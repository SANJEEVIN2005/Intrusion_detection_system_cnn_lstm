"""
test_mobile_remote.py - Unit tests for Mobile SOC Remote Controller and LAN Discovery.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import app, get_local_ip


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_get_local_ip():
    ip = get_local_ip()
    assert isinstance(ip, str)
    assert len(ip) > 6
    # Must be standard IP structure (at least 3 dots)
    parts = ip.split(".")
    assert len(parts) == 4
    for part in parts:
        assert part.isdigit()


def test_mobile_page_route(client):
    response = client.get("/mobile")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "IoT SOC Remote" in html
    assert "Active IPS Auto-Defense" in html
    assert "Remote Attack Simulator" in html


def test_mobile_info_api(client):
    response = client.get("/api/mobile-info")
    assert response.status_code == 200
    data = response.get_json()
    assert "lan_ip" in data
    assert "port" in data
    assert "mobile_url" in data
    assert "/mobile" in data["mobile_url"]
