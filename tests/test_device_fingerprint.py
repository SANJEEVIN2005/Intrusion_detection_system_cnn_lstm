"""
test_device_fingerprint.py - Unit tests for IoT Device Fingerprinting & L7 Application Protocol Dissector (DPI).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from device_fingerprint import (
    resolve_mac_vendor,
    get_device_identity,
    get_application_profile,
    enrich_flow_record,
    MAC_OUI_DATABASE,
    APPLICATION_DATABASE
)


def test_mac_oui_vendor_resolution():
    # 1. Espressif ESP32
    vendor, dev_type, icon = resolve_mac_vendor("24:0A:C4:11:22:33")
    assert "Espressif" in vendor
    assert icon == "📹"

    # 2. Raspberry Pi
    vendor, dev_type, icon = resolve_mac_vendor("B8:27:EB:AA:BB:CC")
    assert "Raspberry Pi" in vendor
    assert icon == "🍓"

    # 3. Apple
    vendor, dev_type, icon = resolve_mac_vendor("F0:18:98:44:55:66")
    assert "Apple" in vendor
    assert icon == "📱"

    # 4. Siemens SCADA
    vendor, dev_type, icon = resolve_mac_vendor("00:0E:8C:99:88:77")
    assert "Siemens" in vendor
    assert icon == "🏭"

    # 5. Unknown MAC
    vendor, dev_type, icon = resolve_mac_vendor("00:00:00:00:00:00")
    assert "Generic" in vendor or "Unknown" in vendor


def test_device_identity_profiling():
    # 1. Localhost
    dev_local = get_device_identity("127.0.0.1")
    assert "Localhost" in dev_local["name"]
    assert dev_local["icon"] == "💻"

    # 2. Gateway Router
    dev_gw = get_device_identity("192.168.1.1")
    assert "Gateway" in dev_gw["name"]
    assert dev_gw["icon"] == "🌐"

    # 3. Public DNS
    dev_dns = get_device_identity("8.8.8.8")
    assert "Google" in dev_dns["name"]

    # 4. Local Subnet Deterministic Device
    dev_node = get_device_identity("192.168.1.45")
    assert len(dev_node["name"]) > 0
    assert len(dev_node["vendor"]) > 0


def test_application_protocol_dissection():
    # 1. MQTT IoT Protocol
    app_mqtt = get_application_profile(1883, "TCP")
    assert "MQTT" in app_mqtt["name"]
    assert app_mqtt["category"] == "IoT Messaging"
    assert app_mqtt["icon"] == "📡"

    # 2. CoAP IoT Protocol
    app_coap = get_application_profile(5683, "UDP")
    assert "CoAP" in app_coap["name"]
    assert app_coap["category"] == "IoT REST"
    assert app_coap["icon"] == "⚡"

    # 3. Modbus SCADA
    app_modbus = get_application_profile(502, "TCP")
    assert "Modbus" in app_modbus["name"]
    assert app_modbus["category"] == "Industrial SCADA"
    assert app_modbus["icon"] == "🏭"

    # 4. RTSP Video Streaming
    app_rtsp = get_application_profile(554, "TCP")
    assert "RTSP" in app_rtsp["name"]
    assert app_rtsp["category"] == "Video Streaming"
    assert app_rtsp["icon"] == "🎥"

    # 5. HTTPS
    app_https = get_application_profile(443, "TCP")
    assert "HTTPS" in app_https["name"]
    assert app_https["icon"] == "🔒"


def test_enrich_flow_record():
    raw_flow = {
        "src_ip": "192.168.1.45",
        "dst_ip": "192.168.1.1",
        "src_port": "52180",
        "dst_port": "1883",
        "protocol": "TCP",
        "predicted_class": "Benign",
        "confidence": 99.8
    }

    enriched = enrich_flow_record(raw_flow)

    assert "src_device_name" in enriched
    assert "dst_device_name" in enriched
    assert "application_name" in enriched
    assert "MQTT" in enriched["application_name"]
    assert "src_icon" in enriched
    assert "dst_icon" in enriched
    assert "application_icon" in enriched
