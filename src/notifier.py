"""
notifier.py

Real-Time Mobile Alert Dispatcher for Telegram and Discord.
Dispatches formatted alert cards to your phone when intrusions are intercepted.
"""

import os
import json
import time
import threading
import urllib.request
import urllib.parse
import urllib.error


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "alerts_config.json")


class ThreatNotifier:
    """Manages Telegram and Discord push notifications with rate limiting."""

    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.last_alert_time = {}
        self.lock = threading.Lock()
        self.config = self._load_config()

    def _load_config(self) -> dict:
        default = {
            "enabled": True,
            "telegram_enabled": False,
            "telegram_bot_token": "",
            "telegram_chat_id": "",
            "discord_enabled": False,
            "discord_webhook_url": "",
            "cooldown_seconds": 5,
        }
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                    default.update(data)
            except Exception as e:
                print(f"[NOTIFIER] Error loading alert config: {e}")

        # Sanitize token & chat ID
        if default.get("telegram_bot_token"):
            default["telegram_bot_token"] = default["telegram_bot_token"].strip().replace(" ", "")
        if default.get("telegram_chat_id"):
            default["telegram_chat_id"] = str(default["telegram_chat_id"]).strip().replace(" ", "")

        return default

    def save_config(self, new_config: dict) -> dict:
        with self.lock:
            # Sanitize inputs
            if "telegram_bot_token" in new_config:
                new_config["telegram_bot_token"] = str(new_config["telegram_bot_token"]).strip().replace(" ", "")
            if "telegram_chat_id" in new_config:
                new_config["telegram_chat_id"] = str(new_config["telegram_chat_id"]).strip().replace(" ", "")
            if "discord_webhook_url" in new_config:
                new_config["discord_webhook_url"] = str(new_config["discord_webhook_url"]).strip()

            self.config.update(new_config)
            try:
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, "w") as f:
                    json.dump(self.config, f, indent=2)
            except Exception as e:
                print(f"[NOTIFIER] Error saving alert config: {e}")
            return self.config

    def get_config(self) -> dict:
        with self.lock:
            cfg = dict(self.config)
            return cfg

    def notify_threat_async(self, flow: dict):
        """Asynchronously dispatches notification to active channels if not on cooldown."""
        if not self.config.get("enabled", True):
            return

        attack_type = flow.get("predicted_class", "Threat")
        if attack_type == "Benign":
            return

        now = time.time()
        cooldown = self.config.get("cooldown_seconds", 5)
        with self.lock:
            last = self.last_alert_time.get(attack_type, 0)
            if now - last < cooldown:
                return  # Cooldown active
            self.last_alert_time[attack_type] = now

        threading.Thread(target=self._dispatch_all, args=(flow,), daemon=True).start()

    def _dispatch_all(self, flow: dict):
        if self.config.get("telegram_enabled") and self.config.get("telegram_bot_token"):
            self.send_telegram_alert(flow)
        if self.config.get("discord_enabled") and self.config.get("discord_webhook_url"):
            self.send_discord_alert(flow)

    def send_telegram_alert(self, flow: dict) -> tuple[bool, str]:
        """Sends rich HTML alert via Telegram Bot API. Returns (success, message)."""
        token = str(self.config.get("telegram_bot_token", "")).strip().replace(" ", "")
        chat_id = str(self.config.get("telegram_chat_id", "")).strip().replace(" ", "")
        if not token or not chat_id:
            return False, "Telegram Bot Token or Chat ID is empty."

        pred_class = flow.get("predicted_class", "Threat")
        conf = flow.get("confidence", 0)
        src = f"{flow.get('src_ip', '?')}:{flow.get('src_port', '?')}"
        dst = f"{flow.get('dst_ip', '?')}:{flow.get('dst_port', '?')}"
        proto = flow.get("protocol", "TCP")
        ts = flow.get("timestamp", time.strftime("%H:%M:%S"))

        src_dev = flow.get("src_device_name", src)
        dst_dev = flow.get("dst_device_name", dst)
        app_name = flow.get("application_name", f"{proto} Traffic")

        text = (
            f"🚨 <b>IoT IDS INTRUSION ALERT!</b> 🚨\n\n"
            f"• <b>Threat:</b> <code>{pred_class} Attack</code>\n"
            f"• <b>Confidence:</b> <code>{conf}%</code>\n"
            f"• <b>Source Device:</b> <code>{src_dev}</code>\n"
            f"• <b>Target Device:</b> <code>{dst_dev}</code>\n"
            f"• <b>Application:</b> <code>{app_name}</code>\n"
            f"• <b>Time:</b> <code>{ts}</code>\n\n"
            f"🛡️ <i>Action: Active IPS Firewall Auto-Mitigation Engaged.</i>"
        )

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode("utf-8")

        try:
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status == 200:
                    return True, "Telegram alert delivered successfully!"
                return False, f"Telegram returned status {resp.status}"
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            print(f"[NOTIFIER] Telegram HTTP {e.code} Error: {err_msg}")
            return False, f"Telegram Error ({e.code}): {err_msg}"
        except Exception as e:
            print(f"[NOTIFIER] Telegram alert exception: {e}")
            return False, f"Telegram Connection Error: {str(e)}"

    def send_discord_alert(self, flow: dict) -> tuple[bool, str]:
        """Sends rich Embed alert via Discord Webhook. Returns (success, message)."""
        webhook_url = str(self.config.get("discord_webhook_url", "")).strip()
        if not webhook_url:
            return False, "Discord Webhook URL is empty."

        pred_class = flow.get("predicted_class", "Threat")
        conf = flow.get("confidence", 0)
        src = f"{flow.get('src_ip', '?')}:{flow.get('src_port', '?')}"
        dst = f"{flow.get('dst_ip', '?')}:{flow.get('dst_port', '?')}"
        proto = flow.get("protocol", "TCP")
        ts = flow.get("timestamp", time.strftime("%H:%M:%S"))

        src_dev = flow.get("src_device_name", src)
        dst_dev = flow.get("dst_device_name", dst)
        app_name = flow.get("application_name", f"{proto} Traffic")

        color = 15548997 if pred_class == "DDoS" else (16345638 if pred_class == "DoS" else 3887350)

        embed = {
            "title": f"🚨 IoT IDS Threat Intercepted: {pred_class}",
            "description": f"Deep Learning Model flagged suspicious network pattern with **{conf}% confidence**.",
            "color": color,
            "fields": [
                {"name": "Attacker Source", "value": f"`{src}`", "inline": True},
                {"name": "Target Destination", "value": f"`{dst}`", "inline": True},
                {"name": "Protocol", "value": f"`{proto}`", "inline": True},
                {"name": "Timestamp", "value": f"`{ts}`", "inline": True},
            ],
            "footer": {"text": "IoT Intrusion Detection System • CNN + LSTM Security Guard"}
        }

        payload = json.dumps({"embeds": [embed]}).encode("utf-8")
        try:
            req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "IoT-IDS-Bot"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status in (200, 204):
                    return True, "Discord alert delivered successfully!"
                return False, f"Discord returned status {resp.status}"
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            print(f"[NOTIFIER] Discord HTTP {e.code} Error: {err_msg}")
            return False, f"Discord Error ({e.code}): {err_msg}"
        except Exception as e:
            print(f"[NOTIFIER] Discord alert exception: {e}")
            return False, f"Discord Connection Error: {str(e)}"

    def send_test_alert(self) -> dict:
        """Dispatches a test alert ping to all configured channels with diagnostic feedback."""
        test_flow = {
            "predicted_class": "DDoS",
            "confidence": 98.5,
            "src_ip": "192.168.1.188",
            "src_port": "54321",
            "dst_ip": "192.168.1.100",
            "dst_port": "443",
            "protocol": "TCP",
            "timestamp": time.strftime("%H:%M:%S")
        }

        results = {}
        if self.config.get("telegram_enabled"):
            tg_ok, tg_msg = self.send_telegram_alert(test_flow)
            results["telegram"] = {"success": tg_ok, "message": tg_msg}

        if self.config.get("discord_enabled"):
            dc_ok, dc_msg = self.send_discord_alert(test_flow)
            results["discord"] = {"success": dc_ok, "message": dc_msg}

        if not self.config.get("telegram_enabled") and not self.config.get("discord_enabled"):
            results["warning"] = "Neither Telegram nor Discord is enabled in the configuration toggle."

        return results
