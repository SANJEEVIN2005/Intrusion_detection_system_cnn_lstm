"""
iot_interface.py

Phase 2 STUB. Same purpose as in Version 1 -- defines the abstract interface
a real live IoT traffic feed would need to implement. Currently returns a
random row from the saved test set as a placeholder.

TODO (Phase 2 - real implementation):
    Replace the body of `get_live_traffic_features()` with one of:
      1. Scapy packet sniffing on a live network interface, extracting
         flow-level features over a sliding time window.
      2. A Raspberry Pi / IoT sensor feed over MQTT or serial.

    The returned vector MUST be in the same order and scale (raw, unscaled)
    as the columns in artifacts/feature_cols.json, since the caller applies
    the saved scaler before feeding it to the model.
"""

import os
import json
import numpy as np


def get_live_traffic_features(artifacts_dir="artifacts", seed=None):
    """Return one 'live' feature vector.

    STUB IMPLEMENTATION: returns a random row from the saved test set.
    Real implementation should replace this with actual packet capture
    (Scapy) or an IoT device feed (MQTT/serial).

    Returns:
        A 1D numpy array of feature values, and the feature column names.
    """
    data = np.load(os.path.join(artifacts_dir, "test_set.npz"))
    X_test = data["X_test"]

    with open(os.path.join(artifacts_dir, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    rng = np.random.default_rng(seed)
    row = X_test[rng.integers(0, len(X_test))]
    return row, feature_cols


if __name__ == "__main__":
    row, cols = get_live_traffic_features()
    print("Simulated live feature vector (stub):")
    for name, val in zip(cols, row):
        print(f"  {name}: {val:.4f}")
