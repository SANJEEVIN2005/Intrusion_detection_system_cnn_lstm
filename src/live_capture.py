"""
live_capture.py

Sniffs live network traffic on a chosen interface (or auto-detected active interface)
using Scapy, aggregates packets into bidirectional flows with full IPv4 + IPv6
and TCP + UDP dual-stack support, extracts CICIDS-compatible statistical features,
and classifies them in real time using the trained PyTorch CNN-LSTM model.

Features:
  1. IPv4 and IPv6 dual-stack: full support for modern mobile hotspots and YouTube/Google IPv6 traffic.
  2. TCP and UDP support: captures HTTPS, HTTP/3, QUIC, and DNS traffic.
  3. Real-time active sweep: flushes completed flows (FIN/RST), idle flows (>=1.5s), and long streams (>=5.0s)
     without waiting for full connection termination.
  4. Auto interface detection: automatically picks the active internet adapter (e.g. "Wi-Fi") if not specified.
  5. Direct in-memory classification: streams classified flows directly to callbacks without file-tailing bottlenecks.
  6. CSV logging: appends classified flows to artifacts/live_flows.csv for logging and auditing.
"""

import argparse
import csv
import json
import os
import pickle
import sys
import threading
import time
import warnings
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6

from cicflowmeter.features.context import PacketDirection
from cicflowmeter.features.flag_count import FlagCount
from cicflowmeter.features.flow_bytes import FlowBytes
from cicflowmeter.features.packet_count import PacketCount
from cicflowmeter.features.packet_length import PacketLength
from cicflowmeter.features.packet_time import PacketTime
from cicflowmeter.utils import get_statistics
import cicflowmeter.constants as constants

sys.path.insert(0, os.path.dirname(__file__))
from model import build_cnn_lstm
from feature_map import live_flow_to_training_vector


# Patch FlowBytes and PacketLength for safe IPv6 and unbuilt Scapy packet handling
def _safe_header_size(self, packet):
    if IP in packet:
        ihl = getattr(packet[IP], "ihl", None) or 5
        return ihl * 4 if TCP in packet else 8
    elif IPv6 in packet:
        return 40 + (packet[TCP].dataofs * 4 if TCP in packet and getattr(packet[TCP], "dataofs", None) else 8)
    return 20
FlowBytes._header_size = _safe_header_size


def _safe_get_initial_ttl(self):
    for packet, _ in self.flow.packets:
        if IP in packet:
            return getattr(packet[IP], "ttl", None) or 64
        elif IPv6 in packet:
            return getattr(packet[IPv6], "hlim", None) or 64
    return 64
FlowBytes.get_initial_ttl = _safe_get_initial_ttl


def _safe_get_header_length(self, packet_direction=None):
    if packet_direction is not None:
        pkts = [p for p, d in self.flow.packets if d == packet_direction]
    else:
        pkts = [p for p, _ in self.flow.packets]
    res = []
    for packet in pkts:
        if IP in packet:
            ihl = getattr(packet[IP], "ihl", None) or 5
            res.append(ihl * 4)
        elif IPv6 in packet:
            res.append(40)
        else:
            res.append(20)
    return res
PacketLength.get_header_length = _safe_get_header_length


def detect_default_interface() -> str:
    """Detects the most appropriate active network interface for sniffing."""
    try:
        iface = scapy.conf.iface
        if iface:
            name = getattr(iface, "name", str(iface))
            if name and name != "Loopback Pseudo-Interface 1":
                return name

        candidates = []
        for iface_obj in scapy.conf.ifaces.values():
            name = getattr(iface_obj, "name", "")
            ip = getattr(iface_obj, "ip", "")
            if "Loopback" in name or not ip or ip.startswith("127."):
                continue
            if "Wi-Fi" in name or "Wireless" in name or "WLAN" in name:
                return name
            candidates.append(name)

        if candidates:
            return candidates[0]
    except Exception:
        pass
    return "Wi-Fi"


def get_packet_flow_key(packet, direction=PacketDirection.FORWARD):
    """Extracts a unique 5-tuple flow key for both IPv4 and IPv6 packets."""
    if TCP in packet:
        protocol = "TCP"
        proto_num = 6
        sport = packet[TCP].sport
        dport = packet[TCP].dport
    elif UDP in packet:
        protocol = "UDP"
        proto_num = 17
        sport = packet[UDP].sport
        dport = packet[UDP].dport
    else:
        return None

    if IP in packet:
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
    elif IPv6 in packet:
        src_ip = packet[IPv6].src
        dst_ip = packet[IPv6].dst
    else:
        return None

    if direction == PacketDirection.FORWARD:
        return (src_ip, dst_ip, sport, dport, proto_num)
    else:
        return (dst_ip, src_ip, dport, sport, proto_num)


class FlowRecord:
    """Represents a single bidirectional network flow and computes CICIDS features."""

    def __init__(self, packet, direction=PacketDirection.FORWARD):
        key = get_packet_flow_key(packet, direction)
        if key is None:
            raise ValueError("Unsupported packet type for flow creation")
        self.src_ip, self.dest_ip, self.src_port, self.dest_port, self.proto_num = key
        self.packets = []
        self.flow_interarrival_time = []
        self.latest_timestamp = 0.0
        self.start_timestamp = 0.0
        self.init_window_size = {PacketDirection.FORWARD: 0, PacketDirection.REVERSE: 0}
        self.start_active = 0.0
        self.last_active = 0.0
        self.active = []
        self.idle = []
        self.forward_bulk_last_timestamp = 0.0
        self.forward_bulk_start_tmp = 0.0
        self.forward_bulk_count = 0
        self.forward_bulk_count_tmp = 0
        self.forward_bulk_duration = 0.0
        self.forward_bulk_packet_count = 0
        self.forward_bulk_size = 0
        self.forward_bulk_size_tmp = 0
        self.backward_bulk_last_timestamp = 0.0
        self.backward_bulk_start_tmp = 0.0
        self.backward_bulk_count = 0
        self.backward_bulk_count_tmp = 0
        self.backward_bulk_duration = 0.0
        self.backward_bulk_packet_count = 0
        self.backward_bulk_size = 0
        self.backward_bulk_size_tmp = 0
        self.protocol = self.proto_num

    @property
    def duration(self) -> float:
        if self.start_timestamp == 0.0:
            return 0.0
        return max(0.0, self.latest_timestamp - self.start_timestamp)

    def add_packet(self, packet, direction: PacketDirection) -> None:
        pkt_time = float(getattr(packet, "time", None) or time.time())
        self.packets.append((packet, direction))
        self.update_flow_bulk(packet, direction)
        self.update_subflow(packet)

        if self.start_timestamp != 0.0:
            self.flow_interarrival_time.append(1e6 * max(0.0, pkt_time - self.latest_timestamp))
        self.latest_timestamp = max(pkt_time, self.latest_timestamp)

        if TCP in packet:
            if direction == PacketDirection.FORWARD and self.init_window_size[direction] == 0:
                self.init_window_size[direction] = getattr(packet[TCP], "window", 0) or 0
            elif direction == PacketDirection.REVERSE:
                self.init_window_size[direction] = getattr(packet[TCP], "window", 0) or 0

        if self.start_timestamp == 0.0:
            self.start_timestamp = pkt_time

    def update_subflow(self, packet):
        pkt_time = float(getattr(packet, "time", None) or time.time())
        last_timestamp = self.latest_timestamp if self.latest_timestamp != 0.0 else pkt_time
        if (pkt_time - last_timestamp) > constants.CLUMP_TIMEOUT:
            self.update_active_idle(pkt_time - last_timestamp)

    def update_active_idle(self, current_time):
        if (current_time - self.last_active) > constants.ACTIVE_TIMEOUT:
            duration = abs(float(self.last_active - self.start_active))
            if duration > 0:
                self.active.append(1e6 * duration)
            self.idle.append(1e6 * (current_time - self.last_active))
            self.start_active = current_time
            self.last_active = current_time
        else:
            self.last_active = current_time

    def update_flow_bulk(self, packet, direction: PacketDirection):
        payload_size = len(PacketCount.get_payload(packet))
        if payload_size == 0:
            return
        pkt_time = float(getattr(packet, "time", None) or time.time())
        if direction == PacketDirection.FORWARD:
            if self.backward_bulk_last_timestamp > self.forward_bulk_start_tmp:
                self.forward_bulk_start_tmp = 0
            if self.forward_bulk_start_tmp == 0:
                self.forward_bulk_start_tmp = pkt_time
                self.forward_bulk_last_timestamp = pkt_time
                self.forward_bulk_count_tmp = 1
                self.forward_bulk_size_tmp = payload_size
            else:
                if (pkt_time - self.forward_bulk_last_timestamp) > constants.CLUMP_TIMEOUT:
                    self.forward_bulk_start_tmp = pkt_time
                    self.forward_bulk_last_timestamp = pkt_time
                    self.forward_bulk_count_tmp = 1
                    self.forward_bulk_size_tmp = payload_size
                else:
                    self.forward_bulk_count_tmp += 1
                    self.forward_bulk_size_tmp += payload_size
                    if self.forward_bulk_count_tmp == constants.BULK_BOUND:
                        self.forward_bulk_count += 1
                        self.forward_bulk_packet_count += self.forward_bulk_count_tmp
                        self.forward_bulk_size += self.forward_bulk_size_tmp
                        self.forward_bulk_duration += (pkt_time - self.forward_bulk_start_tmp)
                    elif self.forward_bulk_count_tmp > constants.BULK_BOUND:
                        self.forward_bulk_packet_count += 1
                        self.forward_bulk_size += payload_size
                        self.forward_bulk_duration += (pkt_time - self.forward_bulk_last_timestamp)
                    self.forward_bulk_last_timestamp = pkt_time
        else:
            if self.forward_bulk_last_timestamp > self.backward_bulk_start_tmp:
                self.backward_bulk_start_tmp = 0
            if self.backward_bulk_start_tmp == 0:
                self.backward_bulk_start_tmp = pkt_time
                self.backward_bulk_last_timestamp = pkt_time
                self.backward_bulk_count_tmp = 1
                self.backward_bulk_size_tmp = payload_size
            else:
                if (pkt_time - self.backward_bulk_last_timestamp) > constants.CLUMP_TIMEOUT:
                    self.backward_bulk_start_tmp = pkt_time
                    self.backward_bulk_last_timestamp = pkt_time
                    self.backward_bulk_count_tmp = 1
                    self.backward_bulk_size_tmp = payload_size
                else:
                    self.backward_bulk_count_tmp += 1
                    self.backward_bulk_size_tmp += payload_size
                    if self.backward_bulk_count_tmp == constants.BULK_BOUND:
                        self.backward_bulk_count += 1
                        self.backward_bulk_packet_count += self.backward_bulk_count_tmp
                        self.backward_bulk_size += self.backward_bulk_size_tmp
                        self.backward_bulk_duration += (pkt_time - self.backward_bulk_start_tmp)
                    elif self.backward_bulk_count_tmp > constants.BULK_BOUND:
                        self.backward_bulk_packet_count += 1
                        self.backward_bulk_size += payload_size
                        self.backward_bulk_duration += (pkt_time - self.backward_bulk_last_timestamp)
                    self.backward_bulk_last_timestamp = pkt_time

    def get_data(self) -> dict:
        flow_bytes = FlowBytes(self)
        flag_count = FlagCount(self)
        packet_count = PacketCount(self)
        packet_length = PacketLength(self)
        packet_time = PacketTime(self)
        flow_iat = get_statistics(self.flow_interarrival_time)
        forward_iat = get_statistics(packet_time.get_packet_iat(PacketDirection.FORWARD))
        backward_iat = get_statistics(packet_time.get_packet_iat(PacketDirection.REVERSE))
        active_stat = get_statistics(self.active)
        idle_stat = get_statistics(self.idle)

        data = {
            "src_ip": self.src_ip,
            "dst_ip": self.dest_ip,
            "src_port": self.src_port,
            "dst_port": self.dest_port,
            "protocol": self.protocol,
            "timestamp": packet_time.get_timestamp(),
            "flow_duration": 1e6 * packet_time.get_duration(),
            "flow_byts_s": flow_bytes.get_rate(),
            "flow_pkts_s": packet_count.get_rate(),
            "fwd_pkts_s": packet_count.get_rate(PacketDirection.FORWARD),
            "bwd_pkts_s": packet_count.get_rate(PacketDirection.REVERSE),
            "tot_fwd_pkts": packet_count.get_total(PacketDirection.FORWARD),
            "tot_bwd_pkts": packet_count.get_total(PacketDirection.REVERSE),
            "totlen_fwd_pkts": packet_length.get_total(PacketDirection.FORWARD),
            "totlen_bwd_pkts": packet_length.get_total(PacketDirection.REVERSE),
            "fwd_pkt_len_max": packet_length.get_max(PacketDirection.FORWARD),
            "fwd_pkt_len_min": packet_length.get_min(PacketDirection.FORWARD),
            "fwd_pkt_len_mean": packet_length.get_mean(PacketDirection.FORWARD),
            "fwd_pkt_len_std": packet_length.get_std(PacketDirection.FORWARD),
            "bwd_pkt_len_max": packet_length.get_max(PacketDirection.REVERSE),
            "bwd_pkt_len_min": packet_length.get_min(PacketDirection.REVERSE),
            "bwd_pkt_len_mean": packet_length.get_mean(PacketDirection.REVERSE),
            "bwd_pkt_len_std": packet_length.get_std(PacketDirection.REVERSE),
            "pkt_len_max": packet_length.get_max(),
            "pkt_len_min": packet_length.get_min(),
            "pkt_len_mean": packet_length.get_mean(),
            "pkt_len_std": packet_length.get_std(),
            "pkt_len_var": packet_length.get_var(),
            "fwd_header_len": flow_bytes.get_forward_header_bytes(),
            "bwd_header_len": flow_bytes.get_reverse_header_bytes(),
            "fwd_seg_size_min": flow_bytes.get_min_forward_header_bytes(),
            "fwd_act_data_pkts": packet_count.has_payload(PacketDirection.FORWARD),
            "flow_iat_mean": flow_iat["mean"],
            "flow_iat_max": flow_iat["max"],
            "flow_iat_min": flow_iat["min"],
            "flow_iat_std": flow_iat["std"],
            "fwd_iat_tot": forward_iat["total"],
            "fwd_iat_max": forward_iat["max"],
            "fwd_iat_min": forward_iat["min"],
            "fwd_iat_mean": forward_iat["mean"],
            "fwd_iat_std": forward_iat["std"],
            "bwd_iat_tot": backward_iat["total"],
            "bwd_iat_max": backward_iat["max"],
            "bwd_iat_min": backward_iat["min"],
            "bwd_iat_mean": backward_iat["mean"],
            "bwd_iat_std": backward_iat["std"],
            "fwd_psh_flags": flag_count.count("PSH", PacketDirection.FORWARD),
            "bwd_psh_flags": flag_count.count("PSH", PacketDirection.REVERSE),
            "fwd_urg_flags": flag_count.count("URG", PacketDirection.FORWARD),
            "bwd_urg_flags": flag_count.count("URG", PacketDirection.REVERSE),
            "fin_flag_cnt": flag_count.count("FIN"),
            "syn_flag_cnt": flag_count.count("SYN"),
            "rst_flag_cnt": flag_count.count("RST"),
            "psh_flag_cnt": flag_count.count("PSH"),
            "ack_flag_cnt": flag_count.count("ACK"),
            "urg_flag_cnt": flag_count.count("URG"),
            "ece_flag_cnt": flag_count.count("ECE"),
            "down_up_ratio": packet_count.get_down_up_ratio(),
            "pkt_size_avg": packet_length.get_avg(),
            "init_fwd_win_byts": self.init_window_size[PacketDirection.FORWARD],
            "init_bwd_win_byts": self.init_window_size[PacketDirection.REVERSE],
            "active_max": active_stat["max"],
            "active_min": active_stat["min"],
            "active_mean": active_stat["mean"],
            "active_std": active_stat["std"],
            "idle_max": idle_stat["max"],
            "idle_min": idle_stat["min"],
            "idle_mean": idle_stat["mean"],
            "idle_std": idle_stat["std"],
            "fwd_byts_b_avg": flow_bytes.get_bytes_per_bulk(PacketDirection.FORWARD),
            "fwd_pkts_b_avg": flow_bytes.get_packets_per_bulk(PacketDirection.FORWARD),
            "bwd_byts_b_avg": flow_bytes.get_bytes_per_bulk(PacketDirection.REVERSE),
            "bwd_pkts_b_avg": flow_bytes.get_packets_per_bulk(PacketDirection.REVERSE),
            "fwd_blk_rate_avg": flow_bytes.get_bulk_rate(PacketDirection.FORWARD),
            "bwd_blk_rate_avg": flow_bytes.get_bulk_rate(PacketDirection.REVERSE),
        }

        data["fwd_seg_size_avg"] = data["fwd_pkt_len_mean"]
        data["bwd_seg_size_avg"] = data["bwd_pkt_len_mean"]
        data["cwr_flag_count"] = data["fwd_urg_flags"]
        data["subflow_fwd_pkts"] = data["tot_fwd_pkts"]
        data["subflow_bwd_pkts"] = data["tot_bwd_pkts"]
        data["subflow_fwd_byts"] = data["totlen_fwd_pkts"]
        data["subflow_bwd_byts"] = data["totlen_bwd_pkts"]
        return data


class LiveFlowManager:
    """Manages active network flows, buffers packets, and flushes flows using an active timer."""

    def __init__(
        self,
        on_flow_callback: Callable[[dict], None],
        idle_timeout: float = 1.5,
        max_duration: float = 4.0,
    ):
        self.flows = {}
        self.lock = threading.Lock()
        self.on_flow_callback = on_flow_callback
        self.idle_timeout = idle_timeout
        self.max_duration = max_duration
        self._stop_event = threading.Event()
        self._sweep_thread = threading.Thread(target=self._sweep_loop, daemon=True)
        self._sweep_thread.start()

    def process_packet(self, pkt):
        """Called by Scapy for every captured packet."""
        if TCP not in pkt and UDP not in pkt:
            return
        if IP not in pkt and IPv6 not in pkt:
            return

        fwd_key = get_packet_flow_key(pkt, PacketDirection.FORWARD)
        rev_key = get_packet_flow_key(pkt, PacketDirection.REVERSE)
        if not fwd_key:
            return

        collected_flow = None
        with self.lock:
            if fwd_key in self.flows:
                flow = self.flows[fwd_key]
                direction = PacketDirection.FORWARD
            elif rev_key in self.flows:
                flow = self.flows[rev_key]
                direction = PacketDirection.REVERSE
            else:
                flow = FlowRecord(pkt, PacketDirection.FORWARD)
                self.flows[fwd_key] = flow
                direction = PacketDirection.FORWARD

            flow.add_packet(pkt, direction)

            # Early collect on TCP FIN or RST flags, or when chunk duration threshold reached
            is_fin_rst = TCP in pkt and bool(set("FR").intersection(pkt[TCP].sprintf("%flags%")))
            is_long = flow.duration >= self.max_duration and len(flow.packets) >= 5

            if is_fin_rst or is_long:
                key_to_del = fwd_key if fwd_key in self.flows else rev_key
                collected_flow = flow
                del self.flows[key_to_del]

        if collected_flow:
            self._emit_flow(collected_flow)

    def _sweep_loop(self):
        """Active background timer to flush idle or timed-out flows immediately."""
        while not self._stop_event.is_set():
            time.sleep(1.0)
            now = time.time()
            to_emit = []
            with self.lock:
                for k in list(self.flows.keys()):
                    flow = self.flows[k]
                    idle_time = now - flow.latest_timestamp
                    if idle_time >= self.idle_timeout or flow.duration >= self.max_duration:
                        to_emit.append(flow)
                        del self.flows[k]
            for flow in to_emit:
                self._emit_flow(flow)

    def _emit_flow(self, flow: FlowRecord):
        try:
            data = flow.get_data()
            if self.on_flow_callback:
                self.on_flow_callback(data)
        except Exception as e:
            print(f"[live_capture] Error preparing flow data: {e}")

    def stop(self):
        self._stop_event.set()
        with self.lock:
            remaining = list(self.flows.values())
            self.flows.clear()
        for flow in remaining:
            self._emit_flow(flow)


class LiveClassifier:
    """Loads the trained model + scaler once, then classifies live flow dicts."""

    def __init__(self, artifacts_dir="artifacts"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(
            os.path.join(artifacts_dir, "cnn_lstm_ids.pt"), map_location=self.device
        )
        self.model = build_cnn_lstm(
            input_dim=checkpoint["input_dim"], num_classes=checkpoint["num_classes"]
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        with open(os.path.join(artifacts_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(artifacts_dir, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)
        with open(os.path.join(artifacts_dir, "feature_cols.json")) as f:
            self.feature_cols = json.load(f)

        from xai_explainer import XAIExplainer
        self.xai = XAIExplainer(artifacts_dir=artifacts_dir)

    def classify(self, live_flow: dict) -> dict:
        """Classifies one live flow dict and returns clean metadata for display."""
        raw_vector = live_flow_to_training_vector(live_flow, self.feature_cols)
        df_input = pd.DataFrame([raw_vector], columns=self.feature_cols)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_scaled = self.scaler.transform(df_input)

        X_tensor = torch.from_numpy(X_scaled.astype(np.float32)).unsqueeze(1).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

        pred_class = self.label_encoder.inverse_transform([pred_idx])[0]

        # Build class probability distribution
        classes = self.label_encoder.classes_
        class_probs = {cls_name: round(probs[i].item() * 100, 1) for i, cls_name in enumerate(classes)}

        # Generate Explainable AI (XAI) feature attributions
        xai_data = self.xai.explain(live_flow, X_scaled, pred_class, round(confidence * 100, 1))

        # Format protocol cleanly
        proto_raw = live_flow.get("protocol", "?")
        if proto_raw in (6, "6", "TCP"):
            proto_str = "TCP"
        elif proto_raw in (17, "17", "UDP"):
            proto_str = "UDP"
        else:
            proto_str = str(proto_raw)

        tot_fwd = int(float(live_flow.get("tot_fwd_pkts", 0) or 0))
        tot_bwd = int(float(live_flow.get("tot_bwd_pkts", 0) or 0))
        duration_us = float(live_flow.get("flow_duration", 0) or 0)

        return {
            "timestamp": time.strftime("%H:%M:%S"),
            "src_ip": str(live_flow.get("src_ip", "?")),
            "dst_ip": str(live_flow.get("dst_ip", "?")),
            "src_port": str(live_flow.get("src_port", "?")),
            "dst_port": str(live_flow.get("dst_port", "?")),
            "protocol": proto_str,
            "predicted_class": pred_class,
            "confidence": round(confidence * 100, 1),
            "class_probs": class_probs,
            "xai": xai_data,
            "metrics": {
                "duration_ms": round(duration_us / 1000.0, 2),
                "total_pkts": tot_fwd + tot_bwd,
                "fwd_pkts": tot_fwd,
                "bwd_pkts": tot_bwd,
                "flow_byts_s": round(float(live_flow.get("flow_byts_s", 0) or 0), 1),
                "flow_pkts_s": round(float(live_flow.get("flow_pkts_s", 0) or 0), 1),
                "pkt_len_mean": round(float(live_flow.get("pkt_len_mean", 0) or 0), 1),
                "pkt_len_max": round(float(live_flow.get("pkt_len_max", 0) or 0), 1),
                "syn_flags": int(float(live_flow.get("syn_flag_cnt", 0) or 0)),
                "fin_flags": int(float(live_flow.get("fin_flag_cnt", 0) or 0)),
                "rst_flags": int(float(live_flow.get("rst_flag_cnt", 0) or 0)),
                "ack_flags": int(float(live_flow.get("ack_flag_cnt", 0) or 0)),
            },
        }


class LiveCaptureRunner:
    """Coordinates packet sniffing, flow aggregation, classification, and output logging."""

    def __init__(
        self,
        iface: Optional[str] = None,
        artifacts_dir: str = "artifacts",
        on_prediction: Optional[Callable[[dict], None]] = None,
        output_csv: str = "artifacts/live_flows.csv",
    ):
        self.iface = iface or detect_default_interface()
        self.artifacts_dir = artifacts_dir
        self.on_prediction = on_prediction
        self.output_csv = output_csv
        self.classifier = LiveClassifier(artifacts_dir=artifacts_dir)
        self.flow_manager = None
        self.sniffer = None
        self._csv_file = None
        self._csv_writer = None
        self._csv_lock = threading.Lock()

    def _init_csv(self):
        try:
            os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
            write_header = not os.path.exists(self.output_csv) or os.path.getsize(self.output_csv) == 0
            self._csv_file = open(self.output_csv, "a", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            if write_header:
                self._csv_writer.writerow([
                    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
                    "protocol", "predicted_class", "confidence"
                ])
                self._csv_file.flush()
        except Exception as e:
            print(f"[live_capture] Notice: CSV logging disabled ({e})")

    def _log_to_csv(self, result: dict):
        if self._csv_writer and self._csv_file:
            with self._csv_lock:
                try:
                    self._csv_writer.writerow([
                        result["timestamp"], result["src_ip"], result["dst_ip"],
                        result["src_port"], result["dst_port"], result["protocol"],
                        result["predicted_class"], result["confidence"]
                    ])
                    self._csv_file.flush()
                except Exception:
                    pass

    def _handle_flow(self, flow_dict: dict):
        try:
            result = self.classifier.classify(flow_dict)
            self._log_to_csv(result)
            print(
                f"[{result['timestamp']}] {result['src_ip']}:{result['src_port']} -> "
                f"{result['dst_ip']}:{result['dst_port']} ({result['protocol']}) | "
                f"{result['predicted_class']} ({result['confidence']}%)"
            )
            if self.on_prediction:
                self.on_prediction(result)
        except Exception as e:
            print(f"[live_capture] Classification error: {e}")

    def start(self):
        self._init_csv()
        self.flow_manager = LiveFlowManager(
            on_flow_callback=self._handle_flow,
            idle_timeout=1.5,
            max_duration=4.0,
        )

        print(f"[LIVE CAPTURE] Starting sniffer on interface '{self.iface}'...")
        self.sniffer = scapy.AsyncSniffer(
            iface=self.iface,
            prn=self.flow_manager.process_packet,
            filter="(ip or ip6) and (tcp or udp)",
            store=False,
        )
        self.sniffer.start()
        print(f"[LIVE CAPTURE] Sniffer active. Capturing IPv4/IPv6 TCP/UDP flows...")

    def stop(self):
        if self.sniffer:
            try:
                self.sniffer.stop()
            except Exception:
                pass
        if self.flow_manager:
            self.flow_manager.stop()
        if self._csv_file:
            try:
                self._csv_file.close()
            except Exception:
                pass
        print("[LIVE CAPTURE] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live IoT Traffic Classifier")
    parser.add_argument("--iface", type=str, default=None,
                        help='Network interface to sniff (e.g. "Wi-Fi"). Auto-detected if omitted.')
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = parser.parse_args()

    runner = LiveCaptureRunner(iface=args.iface, artifacts_dir=args.artifacts_dir)
    runner.start()
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        runner.stop()

