import os
import sys
import time
import pytest
import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from live_capture import (
    get_packet_flow_key,
    FlowRecord,
    LiveFlowManager,
    LiveClassifier,
    detect_default_interface,
)
from cicflowmeter.features.context import PacketDirection


def test_get_packet_flow_key_ipv4_tcp():
    pkt = IP(src="192.168.1.10", dst="93.184.216.34") / TCP(sport=54321, dport=443)
    fwd_key = get_packet_flow_key(pkt, PacketDirection.FORWARD)
    assert fwd_key == ("192.168.1.10", "93.184.216.34", 54321, 443, 6)

    rev_key = get_packet_flow_key(pkt, PacketDirection.REVERSE)
    assert rev_key == ("93.184.216.34", "192.168.1.10", 443, 54321, 6)


def test_get_packet_flow_key_ipv6_udp():
    pkt = IPv6(src="2001:db8::1", dst="2001:db8::2") / UDP(sport=5000, dport=443)
    fwd_key = get_packet_flow_key(pkt, PacketDirection.FORWARD)
    assert fwd_key == ("2001:db8::1", "2001:db8::2", 5000, 443, 17)

    rev_key = get_packet_flow_key(pkt, PacketDirection.REVERSE)
    assert rev_key == ("2001:db8::2", "2001:db8::1", 443, 5000, 17)


def test_flow_record_ipv4_and_ipv6():
    # IPv4 TCP packet
    pkt_v4 = IP(src="192.168.1.50", dst="8.8.8.8") / TCP(sport=12345, dport=80, flags="S")
    flow_v4 = FlowRecord(pkt_v4, PacketDirection.FORWARD)
    flow_v4.add_packet(pkt_v4, PacketDirection.FORWARD)
    data_v4 = flow_v4.get_data()
    assert data_v4["src_ip"] == "192.168.1.50"
    assert data_v4["dst_ip"] == "8.8.8.8"
    assert data_v4["protocol"] == 6
    assert data_v4["tot_fwd_pkts"] == 1

    # IPv6 UDP packet
    pkt_v6 = IPv6(src="2401:4900::1", dst="2001:4860::8888") / UDP(sport=53000, dport=53)
    flow_v6 = FlowRecord(pkt_v6, PacketDirection.FORWARD)
    flow_v6.add_packet(pkt_v6, PacketDirection.FORWARD)
    data_v6 = flow_v6.get_data()
    assert data_v6["src_ip"] == "2401:4900::1"
    assert data_v6["dst_ip"] == "2001:4860::8888"
    assert data_v6["protocol"] == 17
    assert data_v6["tot_fwd_pkts"] == 1


def test_live_flow_manager_flushing():
    emitted = []
    def on_flow(flow_dict):
        emitted.append(flow_dict)

    manager = LiveFlowManager(on_flow_callback=on_flow, idle_timeout=0.5, max_duration=2.0)

    pkt1 = IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1000, dport=80, flags="S")
    pkt1.time = time.time()
    manager.process_packet(pkt1)

    # Let idle timeout trigger
    time.sleep(1.2)
    manager.stop()

    assert len(emitted) >= 1
    assert emitted[0]["src_ip"] == "10.0.0.1"
    assert emitted[0]["dst_ip"] == "10.0.0.2"


def test_live_classifier_inference():
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    if not os.path.exists(os.path.join(artifacts_dir, "cnn_lstm_ids.pt")):
        pytest.skip("Model artifact not found")

    classifier = LiveClassifier(artifacts_dir=artifacts_dir)
    flow_sample = {
        "src_ip": "192.168.1.100",
        "dst_ip": "142.250.190.46",
        "src_port": "54321",
        "dst_port": "443",
        "protocol": "TCP",
        "flow_duration": "50000",
        "tot_fwd_pkts": "10",
        "tot_bwd_pkts": "8",
    }
    result = classifier.classify(flow_sample)
    assert "predicted_class" in result
    assert result["predicted_class"] in ["Benign", "DDoS", "DoS", "Recon"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 100.0


def test_detect_default_interface():
    iface = detect_default_interface()
    assert isinstance(iface, str)
    assert len(iface) > 0
