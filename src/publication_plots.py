"""
publication_plots.py

Generates high-resolution (300 DPI) publication-grade vector figures for research papers:
  1. fig_ablation_comparison.png: 4-panel comparison of 1D-CNN vs CNN-LSTM vs CNN-LSTM-Attention
  2. fig_confusion_matrix.png: Normalized multi-class confusion matrix
  3. fig_roc_pr_curves.png: Multi-Class ROC Curves with AUC
  4. fig_attention_heatmap.png: Self-Attention Weight Heatmap across feature dimensions
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(__file__))
from model import build_cnn_lstm_attention


# High-contrast IEEE publication aesthetic styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})


def plot_ablation_comparison(benchmark_json_path, out_dir="artifacts"):
    """Plots a 4-panel comprehensive performance comparison matching IEEE style."""
    with open(benchmark_json_path, "r") as f:
        data = json.load(f)

    models = ["cnn_baseline", "cnn_lstm", "cnn_lstm_attention"]
    labels = ["1D-CNN [1]", "CNN-LSTM (Paper)", "CNN-LSTM-Attn (Ours)"]
    colors = ["#64748b", "#3b82f6", "#10b981"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # 1. Overall Performance Metrics (Acc, Prec, Rec, F1)
    ax1 = axes[0, 0]
    metrics = ["Accuracy", "Precision", "Recall", "Macro F1"]
    x = np.arange(len(metrics))
    width = 0.25

    for i, m_key in enumerate(models):
        m_data = data[m_key]
        vals = [m_data["accuracy"], m_data["precision_macro"], m_data["recall_macro"], m_data["f1_macro"]]
        bars = ax1.bar(x + (i - 1) * width, vals, width, label=labels[i], color=colors[i], edgecolor="black", linewidth=0.6)
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 2), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7.5, rotation=0)

    ax1.set_title("(a) Overall Metric Comparison", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(80, 103)
    ax1.legend(loc="lower right")
    ax1.grid(True, axis="y")

    # 2. Class-Wise F1 Score Comparison (Highlighting Recon Boost)
    ax2 = axes[0, 1]
    classes = ["Benign", "DDoS", "DoS", "Recon"]
    x2 = np.arange(len(classes))

    for i, m_key in enumerate(models):
        m_data = data[m_key]["class_metrics"]
        vals = [m_data.get(c, {}).get("f1_score", 0.0) for c in classes]
        bars = ax2.bar(x2 + (i - 1) * width, vals, width, label=labels[i], color=colors[i], edgecolor="black", linewidth=0.6)
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f"{height:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 2), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7.5)

    ax2.set_title("(b) Class-wise F1-Score (Highlighting Recon Improvement)", fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes)
    ax2.set_ylabel("F1-Score (%)")
    ax2.set_ylim(75, 104)
    ax2.legend(loc="lower right")
    ax2.grid(True, axis="y")

    # 3. Training Convergence Curves
    ax3 = axes[1, 0]
    for i, m_key in enumerate(models):
        hist = data[m_key].get("history", {})
        val_acc = [v * 100 for v in hist.get("val_acc", [])]
        if val_acc:
            ax3.plot(range(1, len(val_acc) + 1), val_acc, label=f"{labels[i]} (Val)", color=colors[i], marker="o", markersize=4, linewidth=1.5)

    ax3.set_title("(c) Validation Accuracy Convergence per Epoch", fontweight="bold")
    ax3.set_xlabel("Training Epoch")
    ax3.set_ylabel("Validation Accuracy (%)")
    ax3.set_ylim(85, 100)
    ax3.legend(loc="lower right")
    ax3.grid(True)

    # 4. Latency & Throughput Benchmark
    ax4 = axes[1, 1]
    latencies = [data[m_key]["latency_us_per_sample"] for m_key in models]
    throughputs = [data[m_key]["throughput_fps"] for m_key in models]

    ax4_sub = ax4.twinx()
    b1 = ax4.bar(x2[:3] - 0.15, latencies, 0.3, label="Latency (μs/sample)", color="#f59e0b", edgecolor="black", linewidth=0.6)
    b2 = ax4_sub.bar(x2[:3] + 0.15, throughputs, 0.3, label="Throughput (flows/s)", color="#8b5cf6", edgecolor="black", linewidth=0.6)

    ax4.set_title("(d) Edge Deployment Latency & Throughput", fontweight="bold")
    ax4.set_xticks(x2[:3])
    ax4.set_xticklabels(labels, rotation=10)
    ax4.set_ylabel("Latency (μs)", color="#b45309")
    ax4_sub.set_ylabel("Throughput (flows/sec)", color="#6d28d9")
    ax4.grid(True, axis="y")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_ablation_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[PLOTS] Saved '{out_path}'")


def plot_confusion_matrix_heatmap(benchmark_json_path, out_dir="artifacts"):
    """Plots normalized confusion matrix for the proposed CNN-LSTM-Attention model."""
    with open(benchmark_json_path, "r") as f:
        data = json.load(f)

    cm = np.array(data["cnn_lstm_attention"]["confusion_matrix"])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    class_names = data["cnn_lstm_attention"]["class_names"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized Ratio", rotation=-90, va="bottom")

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title="Normalized Confusion Matrix\n(Proposed CNN-LSTM-Attention)",
           ylabel="Ground Truth Class",
           xlabel="Predicted Class")

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_confusion_matrix.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[PLOTS] Saved '{out_path}'")


def plot_attention_heatmaps(artifacts_dir="artifacts"):
    """Visualizes the Self-Attention weights learned across feature sequence tokens."""
    data = np.load(os.path.join(artifacts_dir, "test_set.npz"))
    X_test, y_test = data["X_test"], data["y_test"]

    with open(os.path.join(artifacts_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    checkpoint = torch.load(os.path.join(artifacts_dir, "cnn_lstm_attention_ids.pt"), map_location="cpu")
    model = build_cnn_lstm_attention(checkpoint["input_dim"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    classes = list(label_encoder.classes_)
    fig, axes = plt.subplots(len(classes), 1, figsize=(10, 8), sharex=True)

    for i, cls_name in enumerate(classes):
        indices = np.where(y_test == i)[0]
        if len(indices) == 0:
            continue
        sample_x = torch.from_numpy(X_test[indices[:50]]).unsqueeze(1).float()
        with torch.no_grad():
            _, attn_weights = model(sample_x, return_attention=True)
            mean_attn = attn_weights.mean(dim=0).cpu().numpy()

        ax = axes[i]
        tokens = np.arange(len(mean_attn))
        bars = ax.bar(tokens, mean_attn, color="#0284c7" if cls_name == "Benign" else "#ef4444", edgecolor="black", linewidth=0.5)
        ax.set_title(f"Class: {cls_name} — Attention Weight Distribution $\\alpha_t$", fontweight="bold", fontsize=10)
        ax.set_ylabel("Weight $\\alpha$", fontsize=9)
        ax.set_ylim(0, max(mean_attn) * 1.3)
        ax.grid(True, axis="y")

    axes[-1].set_xlabel("Spatial-Temporal Feature Tokens (Post-Pooling LSTM States)", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(artifacts_dir, "fig_attention_heatmap.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[PLOTS] Saved '{out_path}'")


def generate_all_plots(artifacts_dir="artifacts"):
    json_path = os.path.join(artifacts_dir, "ablation_benchmark_results.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run ablation_study.py first.")
        return

    plot_ablation_comparison(json_path, out_dir=artifacts_dir)
    plot_confusion_matrix_heatmap(json_path, out_dir=artifacts_dir)
    plot_attention_heatmaps(artifacts_dir=artifacts_dir)


if __name__ == "__main__":
    generate_all_plots()
