"""
feature_map.py

The model was trained on CICIDS2017's 80 feature columns (produced by the
original Java CICFlowMeter tool used to build that dataset). For live
capture, we use the `cicflowmeter` Python package, which re-implements the
same flow-feature extraction logic but with different (snake_case) column
names, e.g. "flow_duration" instead of "Flow Duration".

This module maps cicflowmeter's live output field names to the exact
training column names, so a live-captured flow can be fed into the same
trained model without retraining.

IMPORTANT / HONEST LIMITATION:
cicflowmeter and the original CICFlowMeter are two independently-written
tools implementing similar (not byte-for-byte identical) flow statistics.
The mapped values are the closest semantic equivalent, but may differ
slightly in exact computation from what the model saw during training
(e.g. rounding, timeout handling, or bulk-transfer detection edge cases).
This is a well-known practical issue when moving from an offline dataset
to a live capture pipeline, and is worth mentioning explicitly in a demo
or review rather than presenting live predictions as numerically identical
to the paper's methodology.

A handful of training columns have no live equivalent produced by
cicflowmeter (e.g. 'Fwd Header Length.1', a duplicate column quirk in the
original CICIDS2017 CSVs, and 'CWE Flag Count'). These are filled with 0
at inference time -- see UNMAPPED_TRAINING_COLUMNS below.
"""

# training_column_name -> cicflowmeter_live_field_name
LIVE_TO_TRAINING_MAP = {
    "Destination Port": "dst_port",
    "Source Port": "src_port",
    "Protocol": "protocol",
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "tot_fwd_pkts",
    "Total Backward Packets": "tot_bwd_pkts",
    "Total Length of Fwd Packets": "totlen_fwd_pkts",
    "Total Length of Bwd Packets": "totlen_bwd_pkts",
    "Fwd Packet Length Max": "fwd_pkt_len_max",
    "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max",
    "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "flow_byts_s",
    "Flow Packets/s": "flow_pkts_s",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_tot",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_tot",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Bwd PSH Flags": "bwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags",
    "Bwd URG Flags": "bwd_urg_flags",
    "Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",
    "Fwd Packets/s": "fwd_pkts_s",
    "Bwd Packets/s": "bwd_pkts_s",
    "Min Packet Length": "pkt_len_min",
    "Max Packet Length": "pkt_len_max",
    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "Packet Length Variance": "pkt_len_var",
    "FIN Flag Count": "fin_flag_cnt",
    "SYN Flag Count": "syn_flag_cnt",
    "RST Flag Count": "rst_flag_cnt",
    "PSH Flag Count": "psh_flag_cnt",
    "ACK Flag Count": "ack_flag_cnt",
    "URG Flag Count": "urg_flag_cnt",
    "ECE Flag Count": "ece_flag_cnt",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "pkt_size_avg",
    "Avg Fwd Segment Size": "fwd_seg_size_avg",
    "Avg Bwd Segment Size": "bwd_seg_size_avg",
    "Fwd Avg Bytes/Bulk": "fwd_byts_b_avg",
    "Fwd Avg Packets/Bulk": "fwd_pkts_b_avg",
    "Fwd Avg Bulk Rate": "fwd_blk_rate_avg",
    "Bwd Avg Bytes/Bulk": "bwd_byts_b_avg",
    "Bwd Avg Packets/Bulk": "bwd_pkts_b_avg",
    "Bwd Avg Bulk Rate": "bwd_blk_rate_avg",
    "Subflow Fwd Packets": "subflow_fwd_pkts",
    "Subflow Fwd Bytes": "subflow_fwd_byts",
    "Subflow Bwd Packets": "subflow_bwd_pkts",
    "Subflow Bwd Bytes": "subflow_bwd_byts",
    "Init_Win_bytes_forward": "init_fwd_win_byts",
    "Init_Win_bytes_backward": "init_bwd_win_byts",
    "act_data_pkt_fwd": "fwd_act_data_pkts",
    "min_seg_size_forward": "fwd_seg_size_min",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min",
}

# Training columns with no live equivalent from cicflowmeter -- filled with
# 0 at inference time. 'Fwd Header Length.1' is a known duplicate-column
# quirk in the original CICIDS2017 CSVs (identical to 'Fwd Header Length').
# 'CWE Flag Count' has no reliable live equivalent in cicflowmeter's output.
UNMAPPED_TRAINING_COLUMNS = {
    "Fwd Header Length.1": "Fwd Header Length",  # duplicate -> copy value
    "CWE Flag Count": None,  # no live equivalent -> filled with 0
}


def live_flow_to_training_vector(live_flow: dict, feature_cols: list) -> list:
    """Convert a cicflowmeter live flow dict into a feature vector matching
    the exact column order the model was trained on.

    Args:
        live_flow: dict as returned by cicflowmeter's Flow.get_data()
        feature_cols: the ordered list of training column names
            (from artifacts/feature_cols.json)

    Returns:
        A list of floats, one per feature_cols entry, in the same order.
    """
    vector = []
    for col in feature_cols:
        if col in LIVE_TO_TRAINING_MAP:
            live_key = LIVE_TO_TRAINING_MAP[col]
            val = live_flow.get(live_key, 0)
        elif col in UNMAPPED_TRAINING_COLUMNS:
            source_col = UNMAPPED_TRAINING_COLUMNS[col]
            if source_col is None:
                val = 0
            else:
                live_key = LIVE_TO_TRAINING_MAP.get(source_col)
                val = live_flow.get(live_key, 0) if live_key else 0
        else:
            # Should not happen if feature_cols.json matches this mapping,
            # but fail safe rather than crash a live demo.
            val = 0
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        vector.append(val)
    return vector
