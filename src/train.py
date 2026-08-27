"""
train.py

Trains the CNN-LSTM IDS model (PyTorch) on data found in /data, and saves
the trained model + training history + scaler + label encoder to /artifacts.

Usage:
    python src/train.py
    python src/train.py --epochs 5 --batch_size 16
    python src/train.py --max_rows_per_class 20000   (recommended for CICIDS2017)
"""

import argparse
import os
import sys
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import prepare_dataset
from model import build_cnn_lstm


class FlowDataset(Dataset):
    """Wraps preprocessed numpy arrays as a PyTorch Dataset.

    Reshapes each sample to (1, num_features) -- one channel, matching
    what Conv1d expects as input.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).unsqueeze(1)   # (N, 1, num_features)
        self.y = torch.from_numpy(y)                 # (N,) int64 class indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """Run one epoch of training or evaluation. Returns (avg_loss, accuracy)."""
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train CNN-LSTM IoT IDS model (PyTorch)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10, help="paper default: 10")
    parser.add_argument("--batch_size", type=int, default=32, help="paper default: 32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="artifacts")
    parser.add_argument("--max_rows_per_class", type=int, default=None,
                         help="downsample each class to at most this many rows "
                              "(recommended for large real datasets like CICIDS2017, "
                              "e.g. --max_rows_per_class 20000)")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading and preprocessing data from '{args.data_dir}'...")
    X_train, X_test, y_train, y_test, scaler, label_encoder, feature_cols = prepare_dataset(
        data_dir=args.data_dir, seed=args.seed, max_rows_per_class=args.max_rows_per_class
    )

    num_classes = len(label_encoder.classes_)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Class distribution (train): "
          f"{dict(zip(*np.unique(y_train, return_counts=True)))}")

    train_ds = FlowDataset(X_train, y_train)
    test_ds = FlowDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # Train 1. CNN-LSTM (Paper)
    print("\n--- Training Model 1: CNN-LSTM (Ikhlas et al. Paper) ---")
    model_lstm = build_cnn_lstm(input_dim=X_train.shape[1], num_classes=num_classes, seed=args.seed).to(device)
    opt_lstm = torch.optim.Adam(model_lstm.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        run_epoch(model_lstm, train_loader, criterion, opt_lstm, device, train=True)
    val_loss_lstm, val_acc_lstm = run_epoch(model_lstm, test_loader, criterion, opt_lstm, device, train=False)
    print(f"CNN-LSTM Final Val Acc: {val_acc_lstm*100:.2f}%")

    # Train 2. CNN-LSTM-Attention (Proposed Novel)
    print("\n--- Training Model 2: CNN-LSTM-Attention (Proposed Novel) ---")
    from model import build_cnn_lstm_attention, build_cnn_baseline
    model_attn = build_cnn_lstm_attention(input_dim=X_train.shape[1], num_classes=num_classes, seed=args.seed).to(device)
    opt_attn = torch.optim.Adam(model_attn.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        run_epoch(model_attn, train_loader, criterion, opt_attn, device, train=True)
    val_loss_attn, val_acc_attn = run_epoch(model_attn, test_loader, criterion, opt_attn, device, train=False)
    print(f"CNN-LSTM-Attention Final Val Acc: {val_acc_attn*100:.2f}%")

    # Train 3. 1D-CNN Baseline
    print("\n--- Training Model 3: 1D-CNN (Baseline [1]) ---")
    model_base = build_cnn_baseline(input_dim=X_train.shape[1], num_classes=num_classes, seed=args.seed).to(device)
    opt_base = torch.optim.Adam(model_base.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        run_epoch(model_base, train_loader, criterion, opt_base, device, train=True)
    val_loss_base, val_acc_base = run_epoch(model_base, test_loader, criterion, opt_base, device, train=False)
    print(f"1D-CNN Baseline Final Val Acc: {val_acc_base*100:.2f}%")

    # save checkpoints
    torch.save({
        "model_state_dict": model_lstm.state_dict(),
        "input_dim": X_train.shape[1],
        "num_classes": num_classes,
        "feature_cols": feature_cols
    }, os.path.join(args.out_dir, "cnn_lstm_ids.pt"))

    torch.save({
        "model_state_dict": model_attn.state_dict(),
        "input_dim": X_train.shape[1],
        "num_classes": num_classes,
        "feature_cols": feature_cols
    }, os.path.join(args.out_dir, "cnn_lstm_attention_ids.pt"))

    torch.save({
        "model_state_dict": model_base.state_dict(),
        "input_dim": X_train.shape[1],
        "num_classes": num_classes,
        "feature_cols": feature_cols
    }, os.path.join(args.out_dir, "cnn_baseline_ids.pt"))

    with open(os.path.join(args.out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(args.out_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(args.out_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)
    np.savez(os.path.join(args.out_dir, "test_set.npz"), X_test=X_test, y_test=y_test)

    print(f"\nSaved aligned models + test dataset to '{args.out_dir}/'")


if __name__ == "__main__":
    main()
