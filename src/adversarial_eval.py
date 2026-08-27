"""
adversarial_eval.py

Adversarial Robustness & Evasion Defense Benchmark for IEEE Paper Publication.
Evaluates model resilience against mathematical adversarial perturbations:
  1. Fast Gradient Sign Method (FGSM) [Goodfellow et al.]
  2. Projected Gradient Descent (PGD) [Madry et al.]

Compares 3 Architectures:
  - 1D-CNN Baseline (Khan et al. [1])
  - CNN-LSTM (Ikhlas et al., arXiv:2606.05776)
  - CNN-LSTM-Attention (Our Proposed Novel Architecture)

Generates:
  - artifacts/adversarial_benchmark_results.json
  - artifacts/adversarial_table.tex (IEEE formatted LaTeX code)
  - artifacts/fig_adversarial_robustness.png (300 DPI publication curves)
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model import build_cnn_baseline, build_cnn_lstm, build_cnn_lstm_attention


# High-contrast IEEE publication styling
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


def fgsm_attack(model, X, y, epsilon, criterion):
    """Generates Fast Gradient Sign Method (FGSM) adversarial perturbations."""
    if epsilon == 0:
        return X

    X_adv = X.clone().detach().requires_grad_(True)
    logits = model(X_adv)
    loss = criterion(logits, y)
    model.zero_grad()
    loss.backward()

    # Create perturbation: sign of gradient
    data_grad = X_adv.grad.data
    perturbed_X = X_adv + epsilon * data_grad.sign()
    # Clamp to valid normalized feature range [0, 1]
    perturbed_X = torch.clamp(perturbed_X, 0.0, 1.0)
    return perturbed_X.detach()


def pgd_attack(model, X, y, epsilon, alpha=None, num_iter=7, criterion=None):
    """Generates multi-step Projected Gradient Descent (PGD) adversarial perturbations."""
    if epsilon == 0:
        return X

    if alpha is None:
        alpha = epsilon / 4.0

    # Start with random perturbation within epsilon ball
    perturbed_X = X.clone().detach() + torch.FloatTensor(*X.shape).uniform_(-epsilon, epsilon)
    perturbed_X = torch.clamp(perturbed_X, 0.0, 1.0).detach()

    for _ in range(num_iter):
        perturbed_X.requires_grad_(True)
        logits = model(perturbed_X)
        loss = criterion(logits, y)
        model.zero_grad()
        loss.backward()

        data_grad = perturbed_X.grad.data
        perturbed_X = perturbed_X.detach() + alpha * data_grad.sign()

        # Project back into L-infinity epsilon ball around original X
        eta = torch.clamp(perturbed_X - X, min=-epsilon, max=epsilon)
        perturbed_X = torch.clamp(X + eta, min=0.0, max=1.0).detach()

    return perturbed_X


def evaluate_adversarial_robustness(model, test_loader, epsilons, device="cpu"):
    """Evaluates a model across clean and adversarial epsilon budgets."""
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    fgsm_accs = []
    pgd_accs = []

    for eps in epsilons:
        fgsm_correct = 0
        pgd_correct = 0
        total = 0

        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)

            # 1. FGSM Attack
            X_fgsm = fgsm_attack(model, X_b, y_b, eps, criterion)
            with torch.no_grad():
                logits_fgsm = model(X_fgsm)
                fgsm_correct += (logits_fgsm.argmax(dim=1) == y_b).sum().item()

            # 2. PGD Attack
            X_pgd = pgd_attack(model, X_b, y_b, eps, num_iter=7, criterion=criterion)
            with torch.no_grad():
                logits_pgd = model(X_pgd)
                pgd_correct += (logits_pgd.argmax(dim=1) == y_b).sum().item()

            total += X_b.size(0)

        acc_fgsm = round((fgsm_correct / total) * 100, 2)
        acc_pgd = round((pgd_correct / total) * 100, 2)
        fgsm_accs.append(acc_fgsm)
        pgd_accs.append(acc_pgd)
        print(f"  [eps={eps:<4}] -> FGSM Acc: {acc_fgsm:>6.2f}% | PGD Acc: {acc_pgd:>6.2f}%")

    return {"fgsm_accuracy": fgsm_accs, "pgd_accuracy": pgd_accs}


def generate_adversarial_latex_table(results_dict, epsilons):
    """Generates IEEE formatted LaTeX table for adversarial robustness comparison."""
    tex = r"""% =========================================================================
% IEEE Table: Adversarial Evasion Robustness Benchmark (FGSM & PGD Resistance)
% =========================================================================

\begin{table*}[t]
\centering
\caption{Adversarial Robustness Comparison Under Gradient-Based Evasion Attacks}
\label{tab:adversarial_robustness}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{|l|c|c|c|c|c||c|c|c|c|c|}
\hline
\multirow{2}{*}{\textbf{Architecture}} & \multicolumn{5}{c||}{\textbf{FGSM Accuracy (\%) Across Perturbation $\epsilon$}} & \multicolumn{5}{c|}{\textbf{PGD Accuracy (\%) Across Perturbation $\epsilon$}} \\
\cline{2-11}
 & $\epsilon=0.0$ & $\epsilon=0.02$ & $\epsilon=0.05$ & $\epsilon=0.08$ & $\epsilon=0.10$ & $\epsilon=0.0$ & $\epsilon=0.02$ & $\epsilon=0.05$ & $\epsilon=0.08$ & $\epsilon=0.10$ \\
\hline
"""
    model_names = {
        "cnn_baseline": "1D-CNN Baseline [1]",
        "cnn_lstm": "CNN-LSTM (Ikhlas et al.)",
        "cnn_lstm_attention": r"\textbf{CNN-LSTM-Attention (Proposed)}"
    }

    # Extract target indices [0.0, 0.02, 0.05, 0.08, 0.10]
    indices = [0, 2, 3, 4, 5] if len(epsilons) >= 6 else list(range(len(epsilons)))

    for key, name in model_names.items():
        if key not in results_dict:
            continue
        res = results_dict[key]
        fgsm_vals = [res["fgsm_accuracy"][i] for i in indices]
        pgd_vals = [res["pgd_accuracy"][i] for i in indices]

        is_prop = key == "cnn_lstm_attention"
        fmt = lambda v: f"\\textbf{{{v:.1f}\\%}}" if is_prop else f"{v:.1f}\\%"

        fgsm_str = " & ".join(fmt(v) for v in fgsm_vals)
        pgd_str = " & ".join(fmt(v) for v in pgd_vals)
        tex += f"{name} & {fgsm_str} & {pgd_str} \\\\\n"

    tex += r"""\hline
\end{tabular}
\end{table*}
"""
    return tex


def plot_adversarial_curves(results_dict, epsilons, out_dir="artifacts"):
    """Plots publication-grade 300 DPI robustness curves comparing FGSM and PGD defense."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    models = ["cnn_baseline", "cnn_lstm", "cnn_lstm_attention"]
    labels = ["1D-CNN Baseline [1]", "CNN-LSTM (Paper)", "CNN-LSTM-Attention (Ours)"]
    colors = ["#64748b", "#3b82f6", "#10b981"]
    markers = ["s", "^", "o"]

    # 1. FGSM Attack Curve
    for i, m_key in enumerate(models):
        if m_key in results_dict:
            accs = results_dict[m_key]["fgsm_accuracy"]
            lw = 2.2 if m_key == "cnn_lstm_attention" else 1.5
            ax1.plot(epsilons, accs, label=labels[i], color=colors[i], marker=markers[i],
                     linewidth=lw, markersize=5.5)

    ax1.set_title("(a) FGSM Evasion Attack Robustness", fontweight="bold")
    ax1.set_xlabel(r"Perturbation Budget Magnitude ($\epsilon$)")
    ax1.set_ylabel("Detection Accuracy (%)")
    ax1.set_ylim(35, 103)
    ax1.legend(loc="lower left")
    ax1.grid(True)

    # 2. PGD Attack Curve
    for i, m_key in enumerate(models):
        if m_key in results_dict:
            accs = results_dict[m_key]["pgd_accuracy"]
            lw = 2.2 if m_key == "cnn_lstm_attention" else 1.5
            ax2.plot(epsilons, accs, label=labels[i], color=colors[i], marker=markers[i],
                     linewidth=lw, markersize=5.5)

    ax2.set_title("(b) Multi-Step PGD Evasion Attack Robustness", fontweight="bold")
    ax2.set_xlabel(r"Perturbation Budget Magnitude ($\epsilon$)")
    ax2.set_ylabel("Detection Accuracy (%)")
    ax2.set_ylim(30, 103)
    ax2.legend(loc="lower left")
    ax2.grid(True)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_adversarial_robustness.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[ADVERSARIAL] Saved 300 DPI robustness plot to '{out_path}'")


def run_adversarial_benchmark(artifacts_dir="artifacts", batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ADVERSARIAL] Benchmarking on device: {device}")

    # Load test set
    test_path = os.path.join(artifacts_dir, "test_set.npz")
    data = np.load(test_path)
    X_test, y_test = data["X_test"], data["y_test"]

    # Use first 3,000 representative samples for fast high-precision adversarial search
    eval_n = min(3000, len(X_test))
    X_sub = torch.from_numpy(X_test[:eval_n]).unsqueeze(1).float()
    y_sub = torch.from_numpy(y_test[:eval_n]).long()

    test_ds = TensorDataset(X_sub, y_sub)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = X_test.shape[1]
    num_classes = len(np.unique(y_test))

    # Epsilon perturbation budgets
    epsilons = [0.0, 0.01, 0.02, 0.05, 0.08, 0.10]

    # Load trained models
    models = {}

    # 1. CNN-LSTM-Attention
    attn_path = os.path.join(artifacts_dir, "cnn_lstm_attention_ids.pt")
    if os.path.exists(attn_path):
        m_attn = build_cnn_lstm_attention(input_dim, num_classes)
        ckpt = torch.load(attn_path, map_location=device)
        m_attn.load_state_dict(ckpt["model_state_dict"])
        models["cnn_lstm_attention"] = ("CNN-LSTM-Attention (Proposed Novel)", m_attn)

    # 2. Standard CNN-LSTM
    lstm_path = os.path.join(artifacts_dir, "cnn_lstm_ids.pt")
    if os.path.exists(lstm_path):
        m_lstm = build_cnn_lstm(input_dim, num_classes)
        ckpt = torch.load(lstm_path, map_location=device)
        m_lstm.load_state_dict(ckpt["model_state_dict"])
        models["cnn_lstm"] = ("CNN-LSTM (Ikhlas et al. Paper)", m_lstm)

    # 3. 1D-CNN Baseline
    base_path = os.path.join(artifacts_dir, "cnn_baseline_ids.pt")
    if os.path.exists(base_path):
        m_base = build_cnn_baseline(input_dim, num_classes)
        ckpt = torch.load(base_path, map_location=device)
        m_base.load_state_dict(ckpt["model_state_dict"])
        models["cnn_baseline"] = ("1D-CNN (Baseline [1])", m_base)

    results = {}

    for key, (display_name, model_inst) in models.items():
        print(f"\n========================================================")
        print(f" Adversarial Attack Evaluation: {display_name}")
        print(f"========================================================")
        res = evaluate_adversarial_robustness(model_inst, test_loader, epsilons, device=device)
        results[key] = res

    # Save JSON results
    json_path = os.path.join(artifacts_dir, "adversarial_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump({"epsilons": epsilons, "results": results}, f, indent=2)
    print(f"\n[ADVERSARIAL] Saved benchmark JSON to '{json_path}'")

    # Save LaTeX table
    tex_code = generate_adversarial_latex_table(results, epsilons)
    tex_path = os.path.join(artifacts_dir, "adversarial_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_code)
    print(f"[ADVERSARIAL] Saved IEEE LaTeX table to '{tex_path}'")

    # Generate 300 DPI curves
    plot_adversarial_curves(results, epsilons, out_dir=artifacts_dir)

    return results


if __name__ == "__main__":
    run_adversarial_benchmark()
