"""
test_craft_flow.py - Unit tests for Interactive Flow Crafting Studio API and Synthetic Injection.
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_craft_flow_benign(client):
    payload = {
        "src_ip": "192.168.1.50",
        "dst_ip": "192.168.1.1",
        "src_port": 50000,
        "dst_port": 1883,
        "protocol": "TCP",
        "syn_flags": 0,
        "pkts_s": 40,
        "byts_s": 15000
    }
    response = client.post("/api/craft-flow", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "flow" in data
    assert data["flow"]["predicted_class"] == "Benign"
    assert "application_name" in data["flow"]
    assert "MQTT" in data["flow"]["application_name"]


def test_craft_flow_dos_syn_flood(client):
    payload = {
        "src_ip": "192.168.1.199",
        "dst_ip": "192.168.1.1",
        "src_port": 61234,
        "dst_port": 80,
        "protocol": "TCP",
        "syn_flags": 150,
        "pkts_s": 1200,
        "byts_s": 80000
    }
    response = client.post("/api/craft-flow", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["flow"]["predicted_class"] == "DoS"
    assert data["flow"]["metrics"]["syn_flags"] == 150


def test_craft_flow_ddos_burst(client):
    payload = {
        "src_ip": "192.168.1.210",
        "dst_ip": "192.168.1.1",
        "src_port": 58900,
        "dst_port": 554,
        "protocol": "TCP",
        "syn_flags": 0,
        "pkts_s": 3500,
        "byts_s": 250000
    }
    response = client.post("/api/craft-flow", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["flow"]["predicted_class"] == "DDoS"
