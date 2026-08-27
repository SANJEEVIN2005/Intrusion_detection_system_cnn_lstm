import os
import sys
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import prepare_dataset, CLASS_ORDER
from model import build_cnn_lstm


def make_tiny_synthetic_csv(path, n_per_class=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for label in CLASS_ORDER:
        for _ in range(n_per_class):
            rows.append({
                "f1": rng.normal(), "f2": rng.normal(), "f3": rng.normal(),
                "f4": rng.normal(), "f5": rng.normal(), "label": label,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_end_to_end_tiny_dataset():
    tmp_dir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(tmp_dir, "tiny.csv")
        make_tiny_synthetic_csv(csv_path)

        X_train, X_test, y_train, y_test, scaler, label_encoder, feature_cols = \
            prepare_dataset(data_dir=tmp_dir, seed=0)

        X_train_t = torch.from_numpy(X_train).unsqueeze(1)
        X_test_t = torch.from_numpy(X_test).unsqueeze(1)
        y_train_t = torch.from_numpy(y_train)

        model = build_cnn_lstm(input_dim=X_train_t.shape[2], num_classes=4, seed=0)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
        pred_classes = test_logits.argmax(dim=1).numpy()

        assert test_logits.shape[0] == X_test_t.shape[0]
        assert set(pred_classes.tolist()).issubset({0, 1, 2, 3})
    finally:
        shutil.rmtree(tmp_dir)
