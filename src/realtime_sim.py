"""
realtime_sim.py

Phase 1: Simulated real-time inference (PyTorch version). Takes rows
one-by-one from the saved test set, feeds them through the trained model
with a small delay between each, and prints the predicted class.

Usage:
    python src/realtime_sim.py
    python src/realtime_sim.py --n_samples 10 --delay 0.5
"""

import argparse
import os
import sys
import pickle
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import build_cnn_lstm


def run_simulation(artifacts_dir="artifacts", n_samples=10, delay=1.0, seed=42):
    """Run a simulated real-time inference loop over n_samples test rows."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(os.path.join(artifacts_dir, "cnn_lstm_ids.pt"), map_location=device)
    model = build_cnn_lstm(input_dim=checkpoint["input_dim"], num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with open(os.path.join(artifacts_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    data = np.load(os.path.join(artifacts_dir, "test_set.npz"))
    X_test, y_test = data["X_test"], data["y_test"]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)

    print(f"Starting real-time simulation ({len(idx)} samples, {delay}s delay)...\n")
    with torch.no_grad():
        for step, i in enumerate(idx, start=1):
            row = torch.from_numpy(X_test[i]).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,features)
            logits = model(row)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

            pred_class = label_encoder.inverse_transform([pred_idx])[0]
            true_class = label_encoder.inverse_transform([y_test[i]])[0]

            status = "OK" if pred_class == true_class else "MISMATCH"
            print(f"[{step:02d}] incoming flow -> predicted: {pred_class:8s} "
                  f"(confidence {confidence:.2%}) | actual: {true_class:8s} [{status}]")
            time.sleep(delay)

    print("\nSimulation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated real-time IDS inference (PyTorch)")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_simulation(args.artifacts_dir, args.n_samples, args.delay, args.seed)
