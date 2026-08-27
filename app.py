"""
app.py

Flask + Socket.IO web dashboard for the IoT IDS. Shows a live table of
classified network flows as they're captured in real time.

Two modes:
  1. LIVE mode: starts real packet capture (via live_capture.py) on a
     chosen network interface and streams real predictions to the browser.
  2. DEMO mode: replays the saved test set (like realtime_sim.py) so you
     can demo the dashboard UI even without a live capture set up --
     useful for testing the dashboard itself, or as a fallback if Npcap /
     admin privileges aren't available on the demo machine.

Usage:
    python app.py                          # demo mode (safe default)
    python app.py --mode live --iface "Wi-Fi"   # real live capture

Then open http://localhost:5000 in a browser.
"""

import argparse
import json
import os
import pickle
import random
import sys
import threading
import time

import numpy as np
import torch
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from model import build_cnn_lstm
from live_capture import LiveCaptureRunner, detect_default_interface

app = Flask(__name__)
app.config["SECRET_KEY"] = "iot-ids-dashboard"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

ARTIFACTS_DIR = "artifacts"
CURRENT_CONFIG = {
    "mode": "live",
    "iface": None,
}


def run_demo_mode(delay=1.5):
    """Replays the saved test set and emits predictions to the dashboard,
    for demoing the UI without a live capture setup.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(os.path.join(ARTIFACTS_DIR, "cnn_lstm_ids.pt"), map_location=device)
    model = build_cnn_lstm(input_dim=checkpoint["input_dim"], num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with open(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    data = np.load(os.path.join(ARTIFACTS_DIR, "test_set.npz"))
    X_test, y_test = data["X_test"], data["y_test"]

    rng = np.random.default_rng()
    print("[DEMO MODE] Replaying saved test-set flows to the dashboard...")

    while True:
        idx = rng.integers(0, len(X_test))
        row = torch.from_numpy(X_test[idx]).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(row)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

        pred_class = label_encoder.inverse_transform([pred_idx])[0]
        true_class = label_encoder.inverse_transform([y_test[idx]])[0]

        classes = label_encoder.classes_
        class_probs = {cls_name: round(probs[i].item() * 100, 1) for i, cls_name in enumerate(classes)}

        result = {
            "timestamp": time.strftime("%H:%M:%S"),
            "src_ip": f"192.168.1.{rng.integers(2, 250)}",
            "dst_ip": f"104.244.42.{rng.integers(1, 200)}",
            "src_port": str(rng.integers(1024, 65535)),
            "dst_port": "443",
            "protocol": "TCP" if rng.random() > 0.3 else "UDP",
            "predicted_class": pred_class,
            "confidence": round(confidence * 100, 1),
            "true_class": true_class,
            "class_probs": class_probs,
            "metrics": {
                "duration_ms": round(float(rng.uniform(10, 500)), 2),
                "total_pkts": int(rng.integers(5, 50)),
                "fwd_pkts": int(rng.integers(3, 25)),
                "bwd_pkts": int(rng.integers(2, 25)),
                "flow_byts_s": round(float(rng.uniform(500, 150000)), 1),
                "flow_pkts_s": round(float(rng.uniform(5, 120)), 1),
                "pkt_len_mean": round(float(rng.uniform(64, 1400)), 1),
                "pkt_len_max": int(rng.integers(500, 1500)),
                "syn_flags": int(rng.integers(0, 2)),
                "fin_flags": int(rng.integers(0, 2)),
                "rst_flags": int(rng.integers(0, 1)),
                "ack_flags": int(rng.integers(2, 20)),
            },
        }
        socketio.emit("new_flow", result)
        time.sleep(delay)


from notifier import ThreatNotifier
from ips_blocker import IPSBlocker

notifier = ThreatNotifier()
ips_blocker = IPSBlocker(confidence_threshold=90.0, enabled=True)


def run_live_mode(iface):
    """Starts real packet capture and streams real predictions to the dashboard."""
    def on_prediction(result):
        # Auto-block attacker if confidence >= 90%
        blocked = ips_blocker.auto_block_flow(result)
        if blocked:
            result["ips_blocked"] = True
            socketio.emit("ip_blocked", blocked)

        socketio.emit("new_flow", result)
        notifier.notify_threat_async(result)

    runner = LiveCaptureRunner(iface=iface, artifacts_dir=ARTIFACTS_DIR, on_prediction=on_prediction)
    CURRENT_CONFIG["iface"] = runner.iface
    print(f"\n[LIVE MODE] Active sniffing on network interface: '{runner.iface}'")
    print(f"[LIVE MODE] All incoming and outgoing IPv4/IPv6 traffic will update the dashboard in real time.\n")
    runner.start()


def get_local_ip() -> str:
    """Discovers the machine's local LAN IP address on Wi-Fi / Hotspot."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


from attack_simulator import AttackSimulator

simulator = AttackSimulator(artifacts_dir=ARTIFACTS_DIR)


@app.route("/")
def index():
    return render_template("dashboard.html", mode=CURRENT_CONFIG["mode"], iface=CURRENT_CONFIG["iface"])


@app.route("/mobile")
def mobile():
    """Renders the touch-optimized Mobile SOC Remote Controller."""
    return render_template("mobile.html")


@app.route("/api/mobile-info")
def mobile_info():
    lan_ip = get_local_ip()
    port = CURRENT_CONFIG.get("port", 5000)
    return jsonify({
        "lan_ip": lan_ip,
        "port": port,
        "mobile_url": f"http://{lan_ip}:{port}/mobile"
    })


@app.route("/api/status")
def status():
    status_data = dict(CURRENT_CONFIG)
    status_data["ips_enabled"] = ips_blocker.enabled
    status_data["ips_threshold"] = ips_blocker.confidence_threshold
    status_data["blocked_count"] = len(ips_blocker.get_blocked_ips())
    status_data["lan_ip"] = get_local_ip()
    status_data["mobile_url"] = f"http://{get_local_ip()}:{CURRENT_CONFIG.get('port', 5000)}/mobile"
    return jsonify(status_data)


@app.route("/api/ips/status", methods=["GET"])
def ips_status():
    return jsonify({
        "enabled": ips_blocker.enabled,
        "confidence_threshold": ips_blocker.confidence_threshold,
        "blocked_ips": ips_blocker.get_blocked_ips()
    })


@app.route("/api/ips/toggle", methods=["POST"])
def ips_toggle():
    from flask import request
    data = request.get_json() or {}
    enabled = data.get("enabled", not ips_blocker.enabled)
    ips_blocker.set_enabled(enabled)
    return jsonify({"status": "success", "enabled": ips_blocker.enabled})


@app.route("/api/ips/unblock", methods=["POST"])
def ips_unblock():
    from flask import request
    data = request.get_json() or {}
    ip = data.get("ip", "")
    res = ips_blocker.unblock_ip(ip)
    if res.get("success"):
        socketio.emit("ip_unblocked", {"ip": ip})
    return jsonify(res)


@app.route("/api/ips/block", methods=["POST"])
def ips_manual_block():
    from flask import request
    data = request.get_json() or {}
    ip = data.get("ip", "")
    reason = data.get("reason", "Manual Quarantine")
    res = ips_blocker.block_ip(ip, reason=reason, confidence=100.0)
    if res.get("success"):
        socketio.emit("ip_blocked", res["record"])
    return jsonify(res)


@app.route("/api/alerts-config", methods=["GET", "POST"])
def alerts_config():
    from flask import request
    if request.method == "POST":
        data = request.get_json() or {}
        saved = notifier.save_config(data)
        return jsonify({"status": "success", "config": saved})
    return jsonify(notifier.get_config())


@app.route("/api/test-alert", methods=["POST"])
def test_alert():
    res = notifier.send_test_alert()
    return jsonify(res)


@app.route("/api/simulate-attack", methods=["POST"])
def simulate_attack():
    from flask import request
    data = request.get_json() or {}
    attack_type = data.get("type", "dos").lower()
    count = min(int(data.get("count", 1)), 10)

    generated = []
    for _ in range(count):
        if attack_type == "dos":
            flow = simulator.generate_dos_flow()
        elif attack_type == "ddos":
            flow = simulator.generate_ddos_flow()
        elif attack_type == "recon":
            flow = simulator.generate_recon_flow()
        elif attack_type == "burst":
            # Mixed threat burst
            flow = random.choice([
                simulator.generate_dos_flow,
                simulator.generate_ddos_flow,
                simulator.generate_recon_flow
            ])()
        else:
            flow = simulator.generate_dos_flow()

        # Check for automated IPS quarantine on >= 90% confidence
        blocked = ips_blocker.auto_block_flow(flow)
        if blocked:
            flow["ips_blocked"] = True
            socketio.emit("ip_blocked", blocked)

        socketio.emit("new_flow", flow)
        notifier.notify_threat_async(flow)
        generated.append(flow)
        if count > 1:
            time.sleep(0.15)

    return jsonify({"status": "success", "count": len(generated), "flows": generated})


@socketio.on("connect")
def handle_connect():
    socketio.emit("system_status", CURRENT_CONFIG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT IDS Live Web Dashboard")
    parser.add_argument("--mode", choices=["live", "demo"], default="live",
                        help="'live' captures real traffic (default, auto-detects Wi-Fi/adapter); "
                             "'demo' replays test set without packet capture.")
    parser.add_argument("--iface", type=str, default=None,
                        help='Network interface for live mode (e.g. "Wi-Fi"). Auto-detected if omitted.')
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    CURRENT_CONFIG["mode"] = args.mode
    CURRENT_CONFIG["iface"] = args.iface or detect_default_interface()
    CURRENT_CONFIG["port"] = args.port

    if args.mode == "live":
        thread = threading.Thread(target=run_live_mode, args=(args.iface,), daemon=True)
    else:
        thread = threading.Thread(target=run_demo_mode, daemon=True)

    thread.start()

    lan_ip = get_local_ip()
    print("\n" + "=" * 55)
    print(" IoT Intrusion Detection & Prevention System Running")
    print(f" Dashboard URL : http://localhost:{args.port}")
    print(f" Mobile Remote : http://{lan_ip}:{args.port}/mobile")
    print(f" Mode          : {args.mode.upper()}")
    print(f" Interface     : {CURRENT_CONFIG['iface']}")
    print("=" * 55 + "\n")

    socketio.run(app, host="0.0.0.0", port=args.port, debug=False, allow_unsafe_werkzeug=True)
