"""
ips_blocker.py

Active Intrusion Prevention System (IPS) Defense Engine.
Automatically mitigates cyber threats by applying firewall rules to block
attacker IP addresses when AI detection confidence reaches or exceeds 90%.
"""

import os
import json
import time
import subprocess
import threading
import ipaddress


DEFAULT_STORAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "blocked_ips.json")

# Default Whitelist — Never block local host or broadcast
DEFAULT_WHITELIST = {
    "127.0.0.1",
    "::1",
    "localhost",
    "0.0.0.0",
    "255.255.255.255",
}


class IPSBlocker:
    """Manages active IP blocking, Windows Firewall rules, and quarantine pools."""

    def __init__(self, storage_path=DEFAULT_STORAGE_PATH, confidence_threshold=90.0, enabled=True):
        self.storage_path = storage_path
        self.confidence_threshold = float(confidence_threshold)
        self.enabled = enabled
        self.lock = threading.Lock()
        self.blocked_ips = self._load_storage()

    def _load_storage(self) -> dict:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[IPS] Error loading blocked IPs file: {e}")
        return {}

    def _save_storage(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(self.blocked_ips, f, indent=2)
        except Exception as e:
            print(f"[IPS] Error saving blocked IPs file: {e}")

    def is_whitelisted(self, ip: str) -> bool:
        """Checks if an IP address is protected by whitelist."""
        clean_ip = str(ip).strip().lower()
        if clean_ip in DEFAULT_WHITELIST or clean_ip.startswith("127."):
            return True
        try:
            ip_obj = ipaddress.ip_address(clean_ip)
            return ip_obj.is_loopback or ip_obj.is_multicast or ip_obj.is_reserved or ip_obj.is_unspecified
        except ValueError:
            return False

    def block_ip(self, ip: str, reason: str = "Threat Detection", confidence: float = 95.0) -> dict:
        """
        Blocks an attacker IP address using Windows Firewall (netsh) and adds to quarantine pool.
        """
        clean_ip = str(ip).strip()
        if not clean_ip or clean_ip in ("?", "None", "0.0.0.0"):
            return {"success": False, "message": "Invalid IP address."}

        if self.is_whitelisted(clean_ip):
            return {"success": False, "message": f"IP {clean_ip} is whitelisted (loopback/gateway) and cannot be blocked."}

        rule_name = f"IoT_IDS_Block_{clean_ip.replace(':', '_')}"
        firewall_applied = False

        # Attempt to apply Windows Firewall rule
        try:
            cmd = [
                "netsh", "advfirewall", "firewall", "add", "rule",
                f"name={rule_name}",
                "dir=in",
                "action=block",
                f"remoteip={clean_ip}",
                f"description=Auto-blocked by IoT IDS (Threat: {reason}, Conf: {confidence}%)"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 or "Ok." in res.stdout:
                firewall_applied = True
                print(f"[IPS] Successfully added Windows Firewall rule: '{rule_name}'")
            else:
                print(f"[IPS] Firewall command notice: {res.stdout.strip() or res.stderr.strip()}")
        except Exception as e:
            print(f"[IPS] Firewall execution exception (safe mode active): {e}")

        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "ip": clean_ip,
            "rule_name": rule_name,
            "reason": reason,
            "confidence": round(float(confidence), 1),
            "timestamp": now_str,
            "firewall_applied": firewall_applied,
            "status": "QUARANTINED"
        }

        with self.lock:
            self.blocked_ips[clean_ip] = record
            self._save_storage()

        print(f"[IPS DEFENSE] [BLOCKED] Attacker IP '{clean_ip}' has been QUARANTINED! (Reason: {reason} - {confidence}%)")
        return {"success": True, "record": record}

    def unblock_ip(self, ip: str) -> dict:
        """
        Removes an IP address from quarantine and deletes the Windows Firewall rule.
        """
        clean_ip = str(ip).strip()
        with self.lock:
            if clean_ip not in self.blocked_ips:
                return {"success": False, "message": f"IP {clean_ip} was not in quarantine pool."}

            record = self.blocked_ips.pop(clean_ip)
            self._save_storage()

        rule_name = record.get("rule_name", f"IoT_IDS_Block_{clean_ip.replace(':', '_')}")

        # Remove Windows Firewall rule
        try:
            cmd = ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule_name}"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            print(f"[IPS] Removed Windows Firewall rule: '{rule_name}'")
        except Exception as e:
            print(f"[IPS] Exception removing firewall rule: {e}")

        print(f"[IPS DEFENSE] [UNBLOCKED] IP '{clean_ip}' released from quarantine.")
        return {"success": True, "ip": clean_ip, "message": f"IP {clean_ip} successfully released from quarantine."}

    def auto_block_flow(self, flow: dict) -> dict | None:
        """
        Evaluates a flow. If threat confidence >= 90% and IPS is enabled, auto-blocks the source IP.
        """
        if not self.enabled:
            return None

        pred_class = flow.get("predicted_class", "Benign")
        if pred_class == "Benign":
            return None

        conf = float(flow.get("confidence", 0))
        if conf < self.confidence_threshold:
            return None

        src_ip = str(flow.get("src_ip", "")).strip()
        if not src_ip or src_ip in ("?", "None", "0.0.0.0"):
            return None

        # Check if already blocked
        with self.lock:
            if src_ip in self.blocked_ips:
                return None  # Already in quarantine

        # Auto-block attacker
        res = self.block_ip(src_ip, reason=f"{pred_class} Attack", confidence=conf)
        if res.get("success"):
            return res.get("record")
        return None

    def get_blocked_ips(self) -> list[dict]:
        """Returns all currently quarantined IPs."""
        with self.lock:
            return list(self.blocked_ips.values())

    def is_blocked(self, ip: str) -> bool:
        """Checks if an IP is currently blocked."""
        with self.lock:
            return str(ip).strip() in self.blocked_ips

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def set_threshold(self, threshold: float):
        self.confidence_threshold = float(threshold)
