import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import clean_data, encode_labels, scale_features, CLASS_ORDER, _remap_cicids2017_labels


def make_dirty_df():
    df = pd.DataFrame({
        "f1": [1.0, 2.0, np.nan, 4.0, np.inf],
        "f2": [10.0, -np.inf, 30.0, 40.0, 50.0],
        "label": ["Benign", "DDoS", "DoS", "Recon", "Benign"],
    })
    return df


def test_clean_data_removes_nans():
    df = make_dirty_df()
    cleaned = clean_data(df)
    assert not cleaned[["f1", "f2"]].isna().any().any()


def test_clean_data_removes_infs():
    df = make_dirty_df()
    cleaned = clean_data(df)
    numeric = cleaned.drop(columns=["label"])
    assert not np.isinf(numeric.values).any()


def test_clean_data_drops_non_numeric_columns():
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0],
        "flow_id": ["a", "b", "c"],  # non-numeric identifier column
        "label": ["Benign", "DDoS", "DoS"],
    })
    cleaned = clean_data(df)
    assert "flow_id" not in cleaned.columns
    assert "f1" in cleaned.columns and "label" in cleaned.columns


def test_scale_features_in_unit_range():
    df = pd.DataFrame({
        "f1": [0.0, 5.0, 10.0, 20.0],
        "f2": [100.0, 200.0, 300.0, 400.0],
        "label": [0, 1, 2, 3],
    })
    scaled_df, scaler = scale_features(df, label_col="label")
    feats = scaled_df.drop(columns=["label"]).values
    assert feats.min() >= 0.0 - 1e-9
    assert feats.max() <= 1.0 + 1e-9


def test_encode_labels_four_classes():
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4],
        "label": ["Benign", "DDoS", "DoS", "Recon"],
    })
    encoded_df, le = encode_labels(df, "label")
    assert set(encoded_df["label"].unique()) == {0, 1, 2, 3}
    assert list(le.classes_) == CLASS_ORDER


def test_encode_labels_rejects_unknown():
    df = pd.DataFrame({"f1": [1], "label": ["NotARealClass"]})
    with pytest.raises(ValueError):
        encode_labels(df, "label")


def test_cicids2017_label_remap():
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5],
        "label": ["BENIGN", "DDoS", "DoS Hulk", "PortScan", "DoS GoldenEye"],
    })
    remapped = _remap_cicids2017_labels(df, "label")
    assert list(remapped["label"]) == ["Benign", "DDoS", "DoS", "Recon", "DoS"]


def test_prepare_dataset_max_rows_per_class_keeps_label_column():
    # Regression test: groupby().apply() can silently drop the grouping
    # column on newer pandas versions -- this must not happen, since
    # max_rows_per_class relies on the label column surviving downsampling.
    import tempfile
    import shutil
    from preprocess import prepare_dataset

    tmp_dir = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(0)
        rows = []
        for label in CLASS_ORDER:
            for _ in range(50):
                rows.append({"f1": rng.normal(), "f2": rng.normal(), "label": label})
        pd.DataFrame(rows).to_csv(f"{tmp_dir}/tiny.csv", index=False)

        X_train, X_test, y_train, y_test, scaler, le, feature_cols = prepare_dataset(
            data_dir=tmp_dir, seed=0, max_rows_per_class=10
        )
        assert "label" not in feature_cols
        assert X_train.shape[0] + X_test.shape[0] == 40  # 4 classes x 10 rows
    finally:
        shutil.rmtree(tmp_dir)
