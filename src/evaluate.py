"""
evaluate.py

Loads a trained PyTorch model + saved test set and computes accuracy,
precision, recall, F1-score, and a confusion matrix. Also saves plots:
training curves and confusion matrix heatmap.

Usage:
    python src/evaluate.py
"""

import argparse
import os
import sys
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

sys.path.insert(0, os.path.dirname(__file__))
from model import build_cnn_lstm


def plot_history(history, out_dir):
    """Plot train/val accuracy and loss curves per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["accuracy"], label="train_acc")
    axes[0].plot(history["val_accuracy"], label="val_acc")
    axes[0].set_title("Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["loss"], label="train_loss")
    axes[1].plot(history["val_loss"], label="val_loss")
    axes[1].set_title("Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def plot_confusion_matrix(cm, class_names, out_dir):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN-LSTM IoT IDS model (PyTorch)")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = parser.parse_args()
    d = args.artifacts_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(os.path.join(d, "cnn_lstm_ids.pt"), map_location=device)
    model = build_cnn_lstm(input_dim=checkpoint["input_dim"], num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with open(os.path.join(d, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    data = np.load(os.path.join(d, "test_set.npz"))
    X_test, y_test = data["X_test"], data["y_test"]

    X_tensor = torch.from_numpy(X_test).unsqueeze(1).to(device)  # (N, 1, features)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        y_pred = probs.argmax(dim=1).cpu().numpy()

    class_names = list(label_encoder.classes_)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (macro)")
    print(f"Recall:    {rec:.4f} (macro)")
    print(f"F1-score:  {f1:.4f} (macro)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print("Confusion Matrix:")
    print(cm)

    with open(os.path.join(d, "history.json")) as f:
        history = json.load(f)
    plot_history(history, d)
    plot_confusion_matrix(cm, class_names, d)

    metrics = {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1}
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics.json to '{d}/'")


if __name__ == "__main__":
    main()
