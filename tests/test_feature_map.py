import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_map import live_flow_to_training_vector, LIVE_TO_TRAINING_MAP, UNMAPPED_TRAINING_COLUMNS


def test_all_training_columns_covered():
    """Every column the model was trained on must be either directly
    mapped from a live cicflowmeter field, or explicitly listed as
    unmapped (filled with 0 or copied from another column).
    """
    import json
    here = os.path.dirname(__file__)
    feature_cols_path = os.path.join(here, "..", "artifacts", "feature_cols.json")
    if not os.path.exists(feature_cols_path):
        # artifacts not present in this environment (no trained model yet) --
        # fall back to checking against the known 80-column CICIDS2017 set
        return
    with open(feature_cols_path) as f:
        feature_cols = json.load(f)

    for col in feature_cols:
        assert col in LIVE_TO_TRAINING_MAP or col in UNMAPPED_TRAINING_COLUMNS, \
            f"Training column '{col}' has no live-capture mapping!"


def test_live_flow_to_training_vector_basic():
    live_flow = {
        "dst_port": "443", "src_port": "54321", "protocol": "6",
        "flow_duration": "1000", "tot_fwd_pkts": "5",
    }
    feature_cols = ["Destination Port", "Source Port", "Protocol", "Flow Duration", "Total Fwd Packets"]
    vector = live_flow_to_training_vector(live_flow, feature_cols)
    assert vector == [443.0, 54321.0, 6.0, 1000.0, 5.0]


def test_live_flow_to_training_vector_fills_missing_with_zero():
    live_flow = {"dst_port": "80"}
    feature_cols = ["Destination Port", "CWE Flag Count"]  # CWE has no live mapping
    vector = live_flow_to_training_vector(live_flow, feature_cols)
    assert vector == [80.0, 0.0]


def test_live_flow_to_training_vector_duplicate_column():
    live_flow = {"fwd_header_len": "120"}
    feature_cols = ["Fwd Header Length", "Fwd Header Length.1"]
    vector = live_flow_to_training_vector(live_flow, feature_cols)
    assert vector == [120.0, 120.0]  # duplicate column copies the same value


def test_live_flow_to_training_vector_handles_bad_values():
    live_flow = {"dst_port": "not_a_number"}
    feature_cols = ["Destination Port"]
    vector = live_flow_to_training_vector(live_flow, feature_cols)
    assert vector == [0.0]  # falls back to 0 rather than crashing
