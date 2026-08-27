"""
ablation_study.py

Comprehensive Ablation Study & Empirical Benchmarking Suite for IEEE Paper Publication.
Compares three architectures on identical dataset splits:
  1. 1D-CNN (Baseline [1])
  2. CNN-LSTM (Ikhlas et al., arXiv:2606.05776)
  3. CNN-LSTM-Attention (Our Proposed Novel Architecture)

Generates:
  - artifacts/ablation_benchmark_results.json
  - artifacts/paper_tables.tex (IEEE formatted LaTeX code ready for Overleaf)
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score
)

sys.path.insert(0, os.path.dirname(__file__))
from model import build_cnn_baseline, build_cnn_lstm, build_cnn_lstm_attention
from preprocess import prepare_dataset


class FlowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).unsqueeze(1).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            t_loss += loss.item() * X_b.size(0)
            t_correct += (logits.argmax(dim=1) == y_b).sum().item()
            t_total += X_b.size(0)

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                loss = criterion(logits, y_b)
                v_loss += loss.item() * X_b.size(0)
                v_correct += (logits.argmax(dim=1) == y_b).sum().item()
                v_total += X_b.size(0)

        history["train_acc"].append(t_correct / t_total)
        history["val_acc"].append(v_correct / v_total)
        history["train_loss"].append(t_loss / t_total)
        history["val_loss"].append(v_loss / v_total)

    return model, history


def evaluate_model(model, X_test, y_test, label_encoder, device="cpu"):
    model.eval()
    model.to(device)
    X_tensor = torch.from_numpy(X_test).unsqueeze(1).float().to(device)

    # Measure latency
    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()
    end_time = time.perf_counter()

    latency_us_per_sample = ((end_time - start_time) / len(X_test)) * 1e6
    throughput_samples_per_sec = len(X_test) / (end_time - start_time)

    # Compute metrics
    acc = accuracy_score(y_test, preds)
    prec_macro = precision_score(y_test, preds, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, preds, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, preds, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test, preds)
    kappa = cohen_kappa_score(y_test, preds)

    # Class-wise metrics
    class_names = list(label_encoder.classes_)
    prec_classes = precision_score(y_test, preds, average=None, zero_division=0)
    rec_classes = recall_score(y_test, preds, average=None, zero_division=0)
    f1_classes = f1_score(y_test, preds, average=None, zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()

    class_metrics = {}
    for i, name in enumerate(class_names):
        class_metrics[name] = {
            "precision": round(float(prec_classes[i]) * 100, 2),
            "recall": round(float(rec_classes[i]) * 100, 2),
            "f1_score": round(float(f1_classes[i]) * 100, 2),
        }

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "accuracy": round(acc * 100, 2),
        "precision_macro": round(prec_macro * 100, 2),
        "recall_macro": round(rec_macro * 100, 2),
        "f1_macro": round(f1_macro * 100, 2),
        "f1_weighted": round(f1_weighted * 100, 2),
        "mcc": round(mcc, 4),
        "kappa": round(kappa, 4),
        "latency_us_per_sample": round(latency_us_per_sample, 2),
        "throughput_fps": round(throughput_samples_per_sec, 1),
        "param_count": num_params,
        "class_metrics": class_metrics,
        "confusion_matrix": cm,
        "class_names": class_names,
    }


def generate_latex_table(results_dict: dict) -> str:
    """Generates clean, publication-ready IEEE LaTeX table syntax."""
    tex = r"""% =========================================================================
% IEEE Table: Ablation Study Comparison (CNN vs CNN-LSTM vs Proposed CNN-LSTM-Attention)
% Auto-generated for direct inclusion in LaTeX / Overleaf
% =========================================================================

\begin{table*}[t]
\centering
\caption{Ablation Study Performance Comparison on IoT Intrusion Detection}
\label{tab:ablation_comparison}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{|l|c|c|c|c|c|c|c|c|}
\hline
\textbf{Architecture} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{Macro F1} & \textbf{Recon F1} & \textbf{MCC} & \textbf{Params} & \textbf{Latency} \\
\hline
"""
    for model_key, res in results_dict.items():
        name = "1D-CNN (Baseline [1])" if "cnn_baseline" in model_key else (
            "CNN-LSTM (Ikhlas et al.)" if "cnn_lstm" == model_key else
            r"\textbf{CNN-LSTM-Attention (Proposed)}"
        )
        recon_f1 = res["class_metrics"].get("Recon", {}).get("f1_score", 0.0)
        is_prop = "attention" in model_key

        acc_str = f"{res['accuracy']:.2f}\\%"
        prec_str = f"{res['precision_macro']:.2f}\\%"
        rec_str = f"{res['recall_macro']:.2f}\\%"
        f1_str = f"{res['f1_macro']:.2f}\\%"
        recon_str = f"{recon_f1:.2f}\\%"
        mcc_str = f"{res['mcc']:.4f}"

        if is_prop:
            acc_str = f"\\textbf{{{acc_str}}}"
            prec_str = f"\\textbf{{{prec_str}}}"
            rec_str = f"\\textbf{{{rec_str}}}"
            f1_str = f"\\textbf{{{f1_str}}}"
            recon_str = f"\\textbf{{{recon_str}}}"
            mcc_str = f"\\textbf{{{mcc_str}}}"

        p_count = f"{res['param_count']:,}"
        lat_str = f"{res['latency_us_per_sample']:.1f}~\\$\\mu\\$s"

        tex += f"{name} & {acc_str} & {prec_str} & {rec_str} & {f1_str} & {recon_str} & {mcc_str} & {p_count} & {lat_str} \\\\\n"

    tex += r"""\hline
\end{tabular}
\end{table*}
"""
    return tex


def run_ablation(data_dir="data", artifacts_dir="artifacts", epochs=10, batch_size=32, seed=42):
    os.makedirs(artifacts_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ABLATION] Running on device: {device}")

    # Load and preprocess dataset
    print("[ABLATION] Preparing reproducible dataset splits...")
    X_train, X_test, y_train, y_test, scaler, label_encoder, feature_cols = prepare_dataset(
        data_dir=data_dir, seed=seed, max_rows_per_class=15000
    )

    input_dim = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    train_ds = FlowDataset(X_train, y_train)
    test_ds = FlowDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    models_to_test = {
        "cnn_baseline": ("1D-CNN (Baseline [1])", build_cnn_baseline(input_dim, num_classes, seed)),
        "cnn_lstm": ("CNN-LSTM (Ikhlas et al.)", build_cnn_lstm(input_dim, num_classes, seed)),
        "cnn_lstm_attention": ("CNN-LSTM-Attention (Proposed)", build_cnn_lstm_attention(input_dim, num_classes, seed)),
    }

    results = {}
    trained_models = {}

    for model_key, (display_name, model_inst) in models_to_test.items():
        print(f"\n========================================================")
        print(f" Training: {display_name}")
        print(f" Parameters: {sum(p.numel() for p in model_inst.parameters() if p.requires_grad):,}")
        print(f"========================================================")

        trained_m, hist = train_model(model_inst, train_loader, test_loader, epochs=epochs, lr=1e-3, device=device)
        eval_res = evaluate_model(trained_m, X_test, y_test, label_encoder, device=device)
        eval_res["history"] = hist
        results[model_key] = eval_res
        trained_models[model_key] = trained_m

        print(f" -> Accuracy : {eval_res['accuracy']}%")
        print(f" -> Macro F1 : {eval_res['f1_macro']}%")
        print(f" -> Recon F1 : {eval_res['class_metrics'].get('Recon', {}).get('f1_score', '-')}%")
        print(f" -> Latency  : {eval_res['latency_us_per_sample']} us/sample ({eval_res['throughput_fps']} flows/s)")

    # Save trained proposed model checkpoint
    proposed_checkpoint_path = os.path.join(artifacts_dir, "cnn_lstm_attention_ids.pt")
    torch.save({
        "input_dim": input_dim,
        "num_classes": num_classes,
        "model_state_dict": trained_models["cnn_lstm_attention"].state_dict(),
        "feature_cols": feature_cols
    }, proposed_checkpoint_path)
    print(f"\n[ABLATION] Saved proposed model checkpoint to '{proposed_checkpoint_path}'")

    # Save JSON benchmark results
    json_path = os.path.join(artifacts_dir, "ablation_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ABLATION] Saved benchmark JSON to '{json_path}'")

    # Save LaTeX table
    latex_code = generate_latex_table(results)
    tex_path = os.path.join(artifacts_dir, "paper_tables.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"[ABLATION] Saved IEEE LaTeX table to '{tex_path}'")

    return results


if __name__ == "__main__":
    run_ablation()
