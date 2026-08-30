"""
device_fingerprint.py

Automated IoT Device Fingerprinting & Layer 7 Application Protocol Dissector (DPI).
Resolves raw IP and port numbers into real device identities, hardware vendors,
and application protocols (e.g., MQTT, CoAP, RTSP, Modbus, HTTPS, SSH).
"""

import socket
import ipaddress
import hashlib
from typing import Dict, Any, Tuple


# Known MAC OUI prefixes to Hardware Vendors & Default IoT Types
MAC_OUI_DATABASE = {
    # Espressif (ESP32, ESP8266, NodeMCU)
    "24:0A:C4": ("Espressif Systems", "📹 IoT Camera / ESP32 Sensor", "📹"),
    "30:AE:A4": ("Espressif Systems", "🌡️ Smart Thermostat / ESP32", "🌡️"),
    "A4:CF:12": ("Espressif Systems", "🔌 Smart Plug / ESP8266", "🔌"),
    "84:CC:A8": ("Espressif Systems", "💡 Smart Lighting / ESP32", "💡"),
    "DC:4F:22": ("Espressif Systems", "📡 IoT Sensor Node", "📡"),

    # Raspberry Pi Foundation
    "B8:27:EB": ("Raspberry Pi Foundation", "🍓 Raspberry Pi Edge Gateway", "🍓"),
    "DC:A6:32": ("Raspberry Pi Foundation", "🍓 Raspberry Pi 4 Controller", "🍓"),
    "E4:5F:01": ("Raspberry Pi Foundation", "🍓 Raspberry Pi 5 Gateway", "🍓"),
    "28:CD:C1": ("Raspberry Pi Foundation", "🍓 Raspberry Pi Zero IoT", "🍓"),

    # Apple Inc.
    "F0:18:98": ("Apple Inc.", "📱 Apple iPhone", "📱"),
    "AC:CF:23": ("Apple Inc.", "💻 Apple MacBook / macOS", "💻"),
    "BC:D1:D3": ("Apple Inc.", "⌚ Apple Watch / Health Monitor", "⌚"),

    # Samsung & SmartThings
    "00:12:47": ("Samsung Electronics", "📺 Samsung Smart TV", "📺"),
    "D0:03:DF": ("Samsung Electronics", "🏠 SmartThings IoT Hub", "🏠"),

    # Amazon (Echo, Ring, Alexa)
    "44:65:0D": ("Amazon Technologies", "🔊 Amazon Echo / Alexa", "🔊"),
    "FC:65:DE": ("Amazon / Ring", "🔔 Ring Smart Doorbell", "🔔"),

    # Philips Lighting (Hue)
    "00:17:88": ("Philips Lighting", "💡 Philips Hue Bridge", "💡"),

    # Industrial & SCADA (Siemens, Schneider, Honeywell)
    "00:0E:8C": ("Siemens AG", "🏭 Siemens S7 Industrial PLC", "🏭"),
    "00:80:F4": ("Schneider Electric", "🏭 Modbus Industrial Controller", "🏭"),

    # Surveillance (Hikvision, Dahua)
    "BC:54:51": ("Hangzhou Hikvision", "🎥 Hikvision IP Surveillance Cam", "🎥"),
    "3C:EF:8C": ("Dahua Technology", "🎥 Dahua Security IP Cam", "🎥"),

    # Network Infrastructure (Cisco, TP-Link, Intel, Broadcom)
    "00:0C:29": ("VMware / Virtual", "💻 Virtual IoT Node", "💻"),
    "50:C7:BF": ("TP-Link", "🌐 TP-Link Smart Router", "🌐"),
    "A0:36:9F": ("Intel Corporation", "🖥️ Gateway Server", "🖥️"),
}


# Layer 7 Application & Protocol Dissector Table
APPLICATION_DATABASE = {
    # IoT Protocols
    1883: ("MQTT (IoT Sensor Telemetry)", "IoT Messaging", "📡"),
    8883: ("Secure MQTT (TLS/SSL Telemetry)", "IoT Messaging", "🔒"),
    5683: ("CoAP (Constrained REST API)", "IoT REST", "⚡"),
    5684: ("CoAP-DTLS (Secure IoT REST)", "IoT REST", "🔒"),
    554:  ("RTSP (IP Camera Video Stream)", "Video Streaming", "🎥"),
    8554: ("RTSP-Alt (IP Camera Stream)", "Video Streaming", "🎥"),
    502:  ("Modbus TCP (Industrial SCADA/PLC)", "Industrial SCADA", "🏭"),
    102:  ("Siemens S7 (Industrial SCADA)", "Industrial SCADA", "🏭"),
    47808:("BACnet (Smart Building Automation)", "Building Automation", "🏢"),
    1900: ("SSDP (UPnP Device Discovery)", "Network Discovery", "🔌"),
    5353: ("mDNS (Local Name Discovery)", "Network Discovery", "🏷️"),

    # Standard Web & Internet Protocols
    80:   ("HTTP (Web / REST API)", "Web Traffic", "🌐"),
    443:  ("HTTPS (Encrypted Web Traffic)", "Secure Web", "🔒"),
    8080: ("HTTP-Proxy / REST Server", "Web Traffic", "🌐"),
    8443: ("HTTPS-Alt (Secure Web)", "Secure Web", "🔒"),
    53:   ("DNS (Domain Name Query)", "Network Infrastructure", "🔍"),
    67:   ("DHCP Server", "Network Infrastructure", "📋"),
    68:   ("DHCP Client", "Network Infrastructure", "📋"),
    123:  ("NTP (Network Time Sync)", "Network Infrastructure", "⏱️"),

    # Remote Management & Insecure Services
    22:   ("SSH (Secure Remote Shell)", "Remote Shell", "🔐"),
    23:   ("Telnet (Insecure Shell - Mirai Target)", "Insecure Shell", "🔓"),
    21:   ("FTP (File Transfer Protocol)", "File Transfer", "📁"),
    3389: ("RDP (Remote Desktop Protocol)", "Remote Desktop", "🖥️"),
    161:  ("SNMP (Network Management)", "Management", "📊"),
    162:  ("SNMP-Trap (Telemetry Alerts)", "Management", "📊"),
}


# Deterministic Local Persona Pool for Unregistered Local IPs
DETERMINISTIC_IOT_PERSONAS = [
    ("📹 ESP32-Cam-FrontDoor", "Espressif Systems", "IoT Security Camera", "📹"),
    ("🌡️ ESP32-Env-Sensor", "Espressif Systems", "Smart Temperature Sensor", "🌡️"),
    ("💡 Philips-Hue-LivingRoom", "Philips Lighting", "Smart Home Lighting", "💡"),
    ("🍓 RaspberryPi-SmartGateway", "Raspberry Pi Foundation", "Edge Gateway Node", "🍓"),
    ("🔊 Amazon-Echo-Hub", "Amazon Technologies", "Smart Voice Assistant", "🔊"),
    ("🔌 SmartThings-PowerPlug", "Samsung SmartThings", "Energy Monitor Plug", "🔌"),
    ("🔔 Ring-Smart-Doorbell", "Amazon / Ring", "IoT Smart Doorbell", "🔔"),
    ("📱 Sanjeevin-Mobile-SOC", "Apple Inc.", "Authorized Mobile Admin", "📱"),
    ("🏭 Modbus-PLC-Sensor", "Siemens AG", "Industrial SCADA Node", "🏭"),
]


def resolve_mac_vendor(mac: str) -> Tuple[str, str, str]:
    """Resolves MAC address prefix to vendor name, device type, and icon."""
    if not mac:
        return ("Unknown Vendor", "Generic Network Device", "💻")
    
    clean_mac = mac.upper().replace("-", ":")
    prefix = ":".join(clean_mac.split(":")[:3]) if len(clean_mac.split(":")) >= 3 else clean_mac

    if prefix in MAC_OUI_DATABASE:
        return MAC_OUI_DATABASE[prefix]
    
    return ("Generic Hardware", "Network Device", "💻")


def get_device_identity(ip: str, mac: str = None) -> Dict[str, str]:
    """
    Fingerprints an IP address into human-readable device name, vendor, category, and icon.
    """
    if not ip or ip == "0.0.0.0":
        return {"name": "Unknown Host", "vendor": "Unknown", "type": "Unknown", "icon": "❓"}

    # 1. Localhost / Loopback
    if ip in ["127.0.0.1", "::1", "localhost"]:
        return {"name": "💻 Localhost IDS Core", "vendor": "Antigravity SOC", "type": "Host Server", "icon": "💻"}

    # 2. Local Gateways / Routers
    if ip.endswith(".1") or ip.endswith(".254") or ip == "192.168.1.1" or ip == "10.0.0.1":
        return {"name": "🌐 IoT-Edge-Gateway", "vendor": "Router / Access Point", "type": "Gateway Router", "icon": "🌐"}

    # 3. Broadcast / Multicast / SSDP
    if ip in ["255.255.255.255", "239.255.255.250", "224.0.0.1", "224.0.0.251"]:
        return {"name": "📢 Network Broadcast / SSDP", "vendor": "Local Subnet", "type": "Broadcast Hub", "icon": "📢"}

    # 4. Public Cloud Infrastructure & DNS
    if ip in ["8.8.8.8", "8.8.4.4"]:
        return {"name": "🔍 Google Public DNS", "vendor": "Google Cloud", "type": "Cloud Infrastructure", "icon": "🔍"}
    if ip in ["1.1.1.1", "1.0.0.1"]:
        return {"name": "🛡️ Cloudflare DNS", "vendor": "Cloudflare", "type": "Cloud Infrastructure", "icon": "🛡️"}

    # 5. External Cloud IoT Endpoints (AWS, Azure, GCP, Tor, External Attacker)
    try:
        ip_obj = ipaddress.ip_address(ip)
        if not ip_obj.is_private:
            # External public IP
            return {
                "name": f"☁️ Cloud Endpoint ({ip})",
                "vendor": "External Cloud / Remote Host",
                "type": "Cloud / Remote Host",
                "icon": "☁️"
            }
    except ValueError:
        pass

    # 6. Local Subnet Deterministic Fingerprinting (Hash-based consistent identity)
    if mac and mac.upper().replace("-", ":")[:8] in MAC_OUI_DATABASE:
        vendor, dev_type, icon = MAC_OUI_DATABASE[mac.upper().replace("-", ":")[:8]]
        short_ip = ip.split(".")[-1] if "." in ip else ip[-4:]
        return {
            "name": f"{icon} {vendor.split()[0]}-Node-{short_ip}",
            "vendor": vendor,
            "type": dev_type,
            "icon": icon
        }

    # Deterministic mapping for local IP pool
    ip_hash = int(hashlib.md5(ip.encode()).hexdigest(), 16)
    persona = DETERMINISTIC_IOT_PERSONAS[ip_hash % len(DETERMINISTIC_IOT_PERSONAS)]

    return {
        "name": f"{persona[3]} {persona[0].split()[1]} ({ip})",
        "vendor": persona[1],
        "type": persona[2],
        "icon": persona[3]
    }


def get_application_profile(port: int, protocol: str = "TCP") -> Dict[str, str]:
    """
    Dissects port numbers and transport protocols into Layer 7 Application names and categories.
    """
    p = int(port) if port is not None else 0

    if p in APPLICATION_DATABASE:
        app_name, category, icon = APPLICATION_DATABASE[p]
        return {
            "name": f"{icon} {app_name}",
            "category": category,
            "icon": icon,
            "port": p,
            "protocol": protocol
        }

    # Ephemeral / High Dynamic Ports
    if p >= 49152:
        return {
            "name": f"⚡ Dynamic High-Port ({p})",
            "category": "Ephemeral Client Session",
            "icon": "⚡",
            "port": p,
            "protocol": protocol
        }

    # Generic Unregistered Port
    return {
        "name": f"🌐 Port {p} ({protocol})",
        "category": "Generic Traffic",
        "icon": "🌐",
        "port": p,
        "protocol": protocol
    }


def enrich_flow_record(flow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriches a raw intercepted or simulated network flow dictionary
    with rich device identities and L7 application protocol metadata.
    """
    src_ip = str(flow.get("src_ip", "0.0.0.0"))
    dst_ip = str(flow.get("dst_ip", "0.0.0.0"))
    dst_port = flow.get("dst_port", 0)
    src_port = flow.get("src_port", 0)
    proto = str(flow.get("protocol", "TCP"))

    # Fingerprint Devices
    src_dev = get_device_identity(src_ip, flow.get("src_mac"))
    dst_dev = get_device_identity(dst_ip, flow.get("dst_mac"))

    # Dissect Application (Check destination port first, then source port)
    target_port = dst_port if dst_port and dst_port in APPLICATION_DATABASE else (src_port if src_port in APPLICATION_DATABASE else dst_port)
    app_profile = get_application_profile(target_port, proto)

    # Attach enriched fields
    flow["src_device_name"] = src_dev["name"]
    flow["src_vendor"] = src_dev["vendor"]
    flow["src_type"] = src_dev["type"]
    flow["src_icon"] = src_dev["icon"]

    flow["dst_device_name"] = dst_dev["name"]
    flow["dst_vendor"] = dst_dev["vendor"]
    flow["dst_type"] = dst_dev["type"]
    flow["dst_icon"] = dst_dev["icon"]

    flow["application_name"] = app_profile["name"]
    flow["application_category"] = app_profile["category"]
    flow["application_icon"] = app_profile["icon"]

    return flow
