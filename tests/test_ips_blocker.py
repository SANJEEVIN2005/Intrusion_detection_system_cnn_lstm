"""
test_ips_blocker.py - Unit tests for the Active IPS Blocker engine.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ips_blocker import IPSBlocker


def test_ips_blocker_whitelist(tmp_path):
    storage_file = str(tmp_path / "blocked_ips.json")
    ips = IPSBlocker(storage_path=storage_file, confidence_threshold=90.0)

    # Localhost / loopback should never be blocked
    assert ips.is_whitelisted("127.0.0.1") is True
    assert ips.is_whitelisted("::1") is True
    assert ips.is_whitelisted("0.0.0.0") is True

    res = ips.block_ip("127.0.0.1", reason="Test", confidence=99.0)
    assert res["success"] is False
    assert "whitelisted" in res["message"]


def test_ips_block_and_unblock(tmp_path):
    storage_file = str(tmp_path / "blocked_ips.json")
    ips = IPSBlocker(storage_path=storage_file, confidence_threshold=90.0)

    attacker_ip = "192.168.1.188"
    res = ips.block_ip(attacker_ip, reason="DDoS Attack", confidence=98.5)
    assert res["success"] is True
    assert res["record"]["ip"] == attacker_ip
    assert res["record"]["confidence"] == 98.5
    assert ips.is_blocked(attacker_ip) is True

    blocked_list = ips.get_blocked_ips()
    assert len(blocked_list) == 1
    assert blocked_list[0]["ip"] == attacker_ip

    unblock_res = ips.unblock_ip(attacker_ip)
    assert unblock_res["success"] is True
    assert ips.is_blocked(attacker_ip) is False
    assert len(ips.get_blocked_ips()) == 0


def test_ips_auto_block_threshold(tmp_path):
    storage_file = str(tmp_path / "blocked_ips.json")
    ips = IPSBlocker(storage_path=storage_file, confidence_threshold=90.0)

    # Flow with 85% confidence (< 90%) -> should NOT be blocked
    low_conf_flow = {
        "predicted_class": "DoS",
        "confidence": 85.0,
        "src_ip": "192.168.1.200"
    }
    blocked_record = ips.auto_block_flow(low_conf_flow)
    assert blocked_record is None
    assert ips.is_blocked("192.168.1.200") is False

    # Flow with 95% confidence (>= 90%) -> SHOULD be blocked
    high_conf_flow = {
        "predicted_class": "DoS",
        "confidence": 95.0,
        "src_ip": "192.168.1.200"
    }
    blocked_record = ips.auto_block_flow(high_conf_flow)
    assert blocked_record is not None
    assert blocked_record["ip"] == "192.168.1.200"
    assert ips.is_blocked("192.168.1.200") is True
