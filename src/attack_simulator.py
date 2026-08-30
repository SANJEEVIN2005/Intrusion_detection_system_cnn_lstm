"""
attack_simulator.py

Safe, in-application threat generator for demonstration and evaluation purposes.
Uses real attack vectors from the test set for DoS, DDoS, and Recon, runs them
through the CNN-LSTM model, and emits rich, classified threat flows with
realistic dynamic IPs, ports, timestamps, and packet statistics.
"""

import os
import sys
import time
import pickle
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import build_cnn_lstm


class AttackSimulator:
    """Generates authentic threat flows for testing and live evaluation."""

    def __init__(self, artifacts_dir="artifacts", use_quantized: bool = True):
        self.artifacts_dir = artifacts_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_quantized = use_quantized

        # Load proposed CNN-LSTM-Attention model (or fallback to CNN-LSTM)
        attn_path = os.path.join(artifacts_dir, "cnn_lstm_attention_ids.pt")
        base_path = os.path.join(artifacts_dir, "cnn_lstm_ids.pt")

        if os.path.exists(attn_path):
            from model import build_cnn_lstm_attention
            checkpoint = torch.load(attn_path, map_location=self.device)
            self.model = build_cnn_lstm_attention(
                input_dim=checkpoint["input_dim"], num_classes=checkpoint["num_classes"]
            )
        else:
            checkpoint = torch.load(base_path, map_location=self.device)
            self.model = build_cnn_lstm(
                input_dim=checkpoint["input_dim"], num_classes=checkpoint["num_classes"]
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        if self.use_quantized:
            from quantize_model import quantize_network
            self.model = quantize_network(self.model)
            self.model.eval()
            self.precision = "INT8"
        else:
            self.precision = "FP32"

        # Load label encoder
        with open(os.path.join(artifacts_dir, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)

        # Load test set dataset
        test_data = np.load(os.path.join(artifacts_dir, "test_set.npz"))
        self.X_test = test_data["X_test"]
        self.y_test = test_data["y_test"]

        # Cache class index pools
        self.class_indices = {}
        for cls_idx, cls_name in enumerate(self.label_encoder.classes_):
            self.class_indices[cls_name.lower()] = np.where(self.y_test == cls_idx)[0]

        self.rng = np.random.default_rng()

        from xai_explainer import XAIExplainer
        self.xai = XAIExplainer(artifacts_dir=artifacts_dir)

    def _sample_and_classify(self, target_class: str, src_ip=None, dst_ip="192.168.1.100", proto="TCP") -> dict:
        """Pulls a real signature sample of target_class, infers with CNN-LSTM, and formats flow."""
        indices = self.class_indices.get(target_class.lower())
        if indices is None or len(indices) == 0:
            indices = np.arange(len(self.X_test))

        sample_idx = int(self.rng.choice(indices))
        x_raw = self.X_test[sample_idx]
        x_tensor = torch.from_numpy(x_raw).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

        pred_class = self.label_encoder.inverse_transform([pred_idx])[0]
        classes = self.label_encoder.classes_
        class_probs = {cls_name: round(probs[i].item() * 100, 1) for i, cls_name in enumerate(classes)}

        # Realistic metric generation based on attack type
        if target_class.lower() == "dos":
            duration = round(float(self.rng.uniform(15, 120)), 2)
            total_pkts = int(self.rng.integers(80, 250))
            fwd_pkts = total_pkts
            bwd_pkts = 0
            byts_s = round(float(self.rng.uniform(50000, 200000)), 1)
            pkts_s = round(float(self.rng.uniform(1200, 4500)), 1)
            pkt_mean = 40.0
            pkt_max = 40
            syn_flags = total_pkts
            fin_flags = 0
            rst_flags = 0
            ack_flags = 0
        elif target_class.lower() == "ddos":
            duration = round(float(self.rng.uniform(50, 400)), 2)
            total_pkts = int(self.rng.integers(200, 800))
            fwd_pkts = int(total_pkts * 0.9)
            bwd_pkts = total_pkts - fwd_pkts
            byts_s = round(float(self.rng.uniform(200000, 1500000)), 1)
            pkts_s = round(float(self.rng.uniform(3000, 12000)), 1)
            pkt_mean = round(float(self.rng.uniform(800, 1400)), 1)
            pkt_max = 1420
            syn_flags = 0
            fin_flags = 0
            rst_flags = 0
            ack_flags = int(self.rng.integers(10, 50))
        elif target_class.lower() == "recon":
            duration = round(float(self.rng.uniform(1, 10)), 2)
            total_pkts = int(self.rng.integers(2, 4))
            fwd_pkts = total_pkts - 1
            bwd_pkts = 1
            byts_s = round(float(self.rng.uniform(100, 800)), 1)
            pkts_s = round(float(self.rng.uniform(5, 50)), 1)
            pkt_mean = 44.0
            pkt_max = 44
            syn_flags = fwd_pkts
            fin_flags = 0
            rst_flags = 1
            ack_flags = 1
        else:
            duration = round(float(self.rng.uniform(10, 300)), 2)
            total_pkts = int(self.rng.integers(10, 50))
            fwd_pkts = int(total_pkts * 0.6)
            bwd_pkts = total_pkts - fwd_pkts
            byts_s = round(float(self.rng.uniform(1000, 50000)), 1)
            pkts_s = round(float(self.rng.uniform(10, 100)), 1)
            pkt_mean = round(float(self.rng.uniform(100, 800)), 1)
            pkt_max = 1400
            syn_flags = 1
            fin_flags = 1
            rst_flags = 0
            ack_flags = int(total_pkts * 0.8)

        # Generate XAI explanation using authentic vector
        raw_feat_dict = {
            "syn_flag_cnt": syn_flags,
            "flow_pkts_s": pkts_s,
            "flow_byts_s": byts_s,
            "pkt_len_mean": pkt_mean,
            "pkt_len_max": pkt_max,
            "tot_fwd_pkts": fwd_pkts,
            "flow_duration": duration * 1000.0,
            "rst_flag_cnt": rst_flags,
            "ack_flag_cnt": ack_flags,
        }
        xai_data = self.xai.explain(raw_feat_dict, x_raw, pred_class, round(confidence * 100, 1))

        src_port = str(self.rng.integers(1024, 65535))
        dst_port = str(random.choice([80, 443, 8080, 1883, 22, 53]))

        from device_fingerprint import enrich_flow_record
        flow_out = {
            "timestamp": time.strftime("%H:%M:%S"),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": proto,
            "predicted_class": pred_class,
            "confidence": round(confidence * 100, 1),
            "class_probs": class_probs,
            "xai": xai_data,
            "metrics": {
                "duration_ms": duration,
                "total_pkts": total_pkts,
                "fwd_pkts": fwd_pkts,
                "bwd_pkts": bwd_pkts,
                "flow_byts_s": byts_s,
                "flow_pkts_s": pkts_s,
                "pkt_len_mean": pkt_mean,
                "pkt_len_max": pkt_max,
                "syn_flags": syn_flags,
                "fin_flags": fin_flags,
                "rst_flags": rst_flags,
                "ack_flags": ack_flags,
            },
        }
        return enrich_flow_record(flow_out)

    def generate_dos_flow(self, attacker_ip=None, victim_ip="192.168.1.100") -> dict:
        """Simulates a TCP SYN Flood (DoS) flow."""
        attacker_ip = attacker_ip or f"192.168.1.{self.rng.integers(150, 250)}"
        return self._sample_and_classify("dos", src_ip=attacker_ip, dst_ip=victim_ip, proto="TCP")

    def generate_ddos_flow(self, victim_ip="192.168.1.100") -> dict:
        """Simulates a high-volume Distributed Denial of Service (DDoS) flow."""
        bot_ip = f"{self.rng.integers(11, 220)}.{self.rng.integers(1, 250)}.{self.rng.integers(1, 250)}.{self.rng.integers(1, 250)}"
        proto = "UDP" if random.random() > 0.4 else "TCP"
        return self._sample_and_classify("ddos", src_ip=bot_ip, dst_ip=victim_ip, proto=proto)

    def generate_recon_flow(self, scanner_ip=None, victim_ip="192.168.1.100") -> dict:
        """Simulates a PortScan / Reconnaissance probe flow."""
        scanner_ip = scanner_ip or f"192.168.1.{self.rng.integers(180, 220)}"
        return self._sample_and_classify("recon", src_ip=scanner_ip, dst_ip=victim_ip, proto="TCP")
