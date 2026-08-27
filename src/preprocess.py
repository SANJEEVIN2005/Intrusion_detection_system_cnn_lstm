"""
preprocess.py

Generic loading + cleaning + scaling + encoding for CSV-based network flow
traffic data. Two modes are supported:

1. Files already using our 4-class convention (a 'label' column with values
   in {Benign, DDoS, DoS, Recon}) -- these load with zero changes needed.
2. Raw CICIDS2017 CSVs (label column called ' Label', with many fine-grained
   attack names like 'DDoS', 'DoS Hulk', 'DoS GoldenEye', 'PortScan', etc.)
   -- these get remapped automatically via CICIDS2017_LABEL_MAP below.

This means you can drop the real CICIDS2017 CSVs straight into /data with NO
renaming step required.

CONFIRMED WORKING FILE SET (verified against the actual CICIDS2017 release,
"GeneratedLabelledFlows.zip" / TrafficLabelling folder):
    - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv       (BENIGN, DDoS)
    - Wednesday-workingHours.pcap_ISCX.csv                    (BENIGN, DoS Hulk,
                                                                 DoS GoldenEye,
                                                                 DoS slowloris,
                                                                 DoS Slowhttptest,
                                                                 Heartbleed)
    - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv    (BENIGN, PortScan)
NOTE: Friday-WorkingHours-Morning.pcap_ISCX.csv does NOT contain PortScan --
it contains Bot traffic instead, which we don't use. Use the PortScan file
above (Friday afternoon), not the Friday morning file, for the Recon class.
"""

import glob
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

CLASS_ORDER = ["Benign", "DDoS", "DoS", "Recon"]

# Maps CICIDS2017's raw fine-grained attack labels to our 4 classes.
# Keys are matched case-insensitively after stripping whitespace.
CICIDS2017_LABEL_MAP = {
    "benign": "Benign",
    "ddos": "DDoS",
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    "heartbleed": "DoS",
    "portscan": "Recon",
}


def _remap_cicids2017_labels(df, label_col):
    """Remap raw CICIDS2017 attack-name labels to our 4-class convention.

    Any label not found in CICIDS2017_LABEL_MAP is left untouched here --
    encode_labels() will raise a clear error later if anything unmapped
    slips through, rather than silently mislabeling it.
    """
    df = df.copy()
    normalized = df[label_col].astype(str).str.strip().str.lower()
    mapped = normalized.map(CICIDS2017_LABEL_MAP)
    df[label_col] = mapped.where(mapped.notna(), df[label_col].astype(str).str.strip())
    return df


def load_raw_data(data_dir="data"):
    """Load and concatenate all CSV files found in data_dir into one dataframe.

    Handles both our 4-class 'label' column convention and raw CICIDS2017
    files (whose label column is literally named ' Label', with a leading
    space -- a well-known quirk of that dataset's CSVs). Column names are
    stripped of whitespace on load so downstream code only ever sees clean
    names like 'label'.

    Raises a clear error if the directory is missing or has no CSVs.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory '{data_dir}' does not exist. "
            f"Place your CSVs (e.g. CICIDS2017 files) in '{data_dir}'."
        )
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. "
            f"Place your CSVs (e.g. CICIDS2017 files) in '{data_dir}'."
        )

    dfs = []
    for f in csv_files:
        d = pd.read_csv(f, low_memory=False)
        d.columns = [c.strip() for c in d.columns]  # strips ' Label' -> 'Label'
        if "Label" in d.columns and "label" not in d.columns:
            d = d.rename(columns={"Label": "label"})
            d = _remap_cicids2017_labels(d, "label")
        dfs.append(d)

    # Align columns across files (CICIDS2017 files sometimes have minor
    # column-set differences); keep only columns common to all files.
    common_cols = set(dfs[0].columns)
    for d in dfs[1:]:
        common_cols &= set(d.columns)
    common_cols = list(common_cols)
    dfs = [d[common_cols] for d in dfs]

    df = pd.concat(dfs, ignore_index=True)
    return df


def clean_data(df):
    """Remove missing values and handle infinite values.

    - Replaces +/-inf with NaN, then drops all rows containing NaN.
    - Drops any non-numeric columns except 'label' (CICIDS2017 has a few
      identifier-like columns that aren't useful numeric features).
    - Returns a cleaned copy of the dataframe (index reset).
    """
    df = df.copy()

    # drop non-numeric feature columns (keep label)
    keep_cols = ["label"] + [
        c for c in df.columns
        if c != "label" and pd.api.types.is_numeric_dtype(df[c])
    ]
    df = df[keep_cols]

    numeric_cols = [c for c in df.columns if c != "label"]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


def encode_labels(df, label_col="label"):
    """Encode the label column into 4 integer classes using a fixed class order.

    Returns (df_with_int_labels, label_encoder). Any label not in CLASS_ORDER
    raises a ValueError so silent mislabeling is never possible -- this is
    what catches any CICIDS2017 attack type we haven't explicitly mapped.
    """
    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.strip()

    unknown = set(df[label_col].unique()) - set(CLASS_ORDER)
    if unknown:
        raise ValueError(
            f"Found labels not in expected classes {CLASS_ORDER}: {unknown}. "
            f"If these are raw CICIDS2017 attack names, add them to "
            f"CICIDS2017_LABEL_MAP in preprocess.py."
        )

    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    df[label_col] = le.transform(df[label_col])
    return df, le


def scale_features(df, label_col="label", scaler=None):
    """Scale all non-label numeric columns to [0, 1] using MinMaxScaler.

    If a fitted scaler is passed, it is reused (for consistent test-set scaling).
    Returns (scaled_df, fitted_scaler).
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c != label_col]
    if scaler is None:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    return df, scaler


def prepare_dataset(data_dir="data", label_col="label", test_size=0.2, seed=42,
                     max_rows_per_class=None):
    """Full preprocessing pipeline: load -> clean -> encode -> scale -> split.

    NOTE: an 80/20 stratified train/test split is used, matching Version 1.

    Args:
        max_rows_per_class: if set, randomly downsamples each class to at
            most this many rows after cleaning. Useful for CICIDS2017 since
            Benign traffic vastly outnumbers attack traffic and full files
            can be tens of millions of rows -- this keeps training fast on
            a laptop CPU/GPU while preserving class balance.

    Returns: X_train, X_test, y_train, y_test, scaler, label_encoder, feature_cols
    """
    df = load_raw_data(data_dir)
    df = clean_data(df)

    if max_rows_per_class is not None:
        counts = df[label_col].value_counts()
        parts = []
        for cls, n in counts.items():
            n_take = min(n, max_rows_per_class)
            parts.append(df[df[label_col] == cls].sample(n=n_take, random_state=seed))
        df = pd.concat(parts, ignore_index=True)

    df, label_encoder = encode_labels(df, label_col)

    feature_cols = [c for c in df.columns if c != label_col]
    df, scaler = scale_features(df, label_col)

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler, label_encoder, feature_cols
