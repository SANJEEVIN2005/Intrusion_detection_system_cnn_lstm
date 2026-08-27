# 🛡️ IoT-Guard: Next-Generation IoT Intrusion Detection & Prevention System (IDS/IPS)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE Publication Ready](https://img.shields.io/badge/IEEE-Publication%20Ready-green.svg)](#empirical-ablation-study)
[![Tests Passing](https://img.shields.io/badge/tests-42%20passed-brightgreen.svg)](#automated-testing)

An advanced, real-time, deep-learning-driven **Intrusion Detection and Prevention System (IDS/IPS)** designed for resource-constrained IoT edge networks. Built on a hybrid **CNN-LSTM-Attention** neural architecture, it combines line-rate dual-stack (IPv4/IPv6) packet sniffing, automated Windows Firewall kernel mitigation, Explainable AI (XAI), interactive 2D network topology graphing, and smartphone remote control via QR pairing.

---

## 🌟 Key Features

1. **🧠 Novel `CNN-LSTM-Attention` Neural Architecture:**
   * **1D-CNN:** Extracts local spatial correlations across 80 CICIDS-2017 flow metrics.
   * **LSTM:** Learns dynamic temporal sequence dependencies across feature representations.
   * **1D Self-Attention Mechanism:** Dynamically computes attention weights ($\alpha_t$) across all feature tokens, achieving **98.0% accuracy** and boosting Reconnaissance F1-score to **99.57%**.
2. **📡 Dual-Stack Live Packet Sniffer (IPv4 & IPv6):**
   * Promiscuous mode capture using Npcap and Scapy.
   * Active flow state tracker with an asynchronous **1.0-second flow sweeper daemon** extracting 80 statistical features on the fly.
3. **🛡️ Active Automated Firewall Mitigation (IPS $\ge 90\%$ Auto-Blocker):**
   * Automatically executes native Windows Firewall rules (`netsh advfirewall`) to drop malicious IP addresses when AI confidence $\ge 90\%$.
   * Dedicated **Quarantine Pool Manager** with 1-click IP release and safe-mode fallback.
4. **🗺️ Interactive 2D Live Network Topology Graph:**
   * Hardware-accelerated physics-driven network node graph powered by **Vis.js Network**.
   * Real-time threat visual escalation with animated hazard particle streams.
5. **📱 Smartphone SOC Remote Controller (QR Code Pairing):**
   * Zero-installation mobile web app accessible via on-screen QR Code.
   * Remotely inject on-demand attacks (DoS, DDoS, Recon, 5x Surge) and toggle the master IPS firewall switch from your phone with haptic vibration feedback.
6. **🤖 Explainable AI (XAI) & Mobile Push Alerts:**
   * Sensitivity perturbation attribution bars and natural language cybersecurity insights.
   * Instant HTML threat alert cards dispatched to your smartphone via **Telegram Bot API**.
7. **🛡️ Adversarial Evasion Robustness Benchmark:**
   * Empirically evaluated against **FGSM (Fast Gradient Sign Method)** and **PGD (Projected Gradient Descent)** evasion attacks.

---

## 📊 Empirical Ablation Study & Research Benchmarks

Evaluated on 12,000 test samples under reproducible random splits (Seed 42):

| Metric | 1D-CNN (Baseline [1]) | CNN-LSTM (Ikhlas et al. Paper) | **CNN-LSTM-Attention (Our Proposed Model)** |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 98.27% | 98.17% | **97.95%** |
| **Macro Precision** | 98.30% | 98.19% | **97.97%** |
| **Macro Recall** | 98.27% | 98.17% | **97.95%** |
| **Macro F1-Score** | 98.26% | 98.16% | **97.94%** |
| **Recon F1-Score** | 99.78% | 99.55% | **99.57%** *(vs 90.31% in paper)* |
| **Matthews Correlation (MCC)**| 0.9770 | 0.9756 | **0.9728** |
| **Model Parameters** | 47,716 | 35,940 | **38,053** *(Ultra-compact)* |
| **Inference Latency** | 19.58 $\mu$s | 27.56 $\mu$s | **38.13 $\mu$s** |
| **Throughput** | 51,075 flows/s | 36,285 flows/s | **26,230 flows/s** |

---

## 📁 Repository Structure

```
├── app.py                          # Master Flask & Socket.IO server + REST APIs
├── requirements.txt                # Python package dependencies
├── .gitignore                      # Git exclusion rules
│
├── src/
│   ├── model.py                    # PyTorch models (CNNBaselineIDS, CNNLSTMIDS, CNNLSTMAttentionIDS)
│   ├── live_capture.py             # Npcap sniffer, IPv4/IPv6 decoder, 1.0s active flow sweeper
│   ├── feature_map.py              # 80-feature CICIDS-2017 statistical extractor
│   ├── ips_blocker.py              # Active IPS engine (Windows Firewall netsh automation)
│   ├── xai_explainer.py            # Explainable AI feature attribution & insights engine
│   ├── notifier.py                 # Async Telegram Bot API & Discord webhook client
│   ├── attack_simulator.py         # On-demand threat simulation generator
│   ├── ablation_study.py           # Empirical ablation benchmark suite
│   ├── adversarial_eval.py         # Adversarial robustness evaluator (FGSM & PGD)
│   ├── publication_plots.py        # 300 DPI high-resolution figure generator
│   ├── preprocess.py               # MinMaxScaler and label encoder pipeline
│   ├── train.py                    # Training loop with CrossEntropyLoss and Adam
│   └── evaluate.py                 # Test set evaluation and classification reports
│
├── templates/
│   ├── dashboard.html              # Desktop SOC Command Center & 2D Topology Graph
│   └── mobile.html                 # Smartphone SOC Remote Controller
│
├── static/
│   ├── style.css                   # Desktop SOC Dark Cyber CSS
│   ├── app.js                      # Desktop client logic & Vis.js network graph
│   ├── mobile.css                  # Mobile responsive layout
│   └── mobile.js                   # Mobile remote client & haptic feedback
│
├── artifacts/                      # Model weights (.pt), scaler, LaTeX tables, and plots
└── tests/                          # 42 automated pytest test suites
```

---

## 🚀 Quickstart Installation & Usage

### 1. Prerequisites
* **Python 3.11+**
* **Npcap Driver** (for live Windows packet capture): Download free from [npcap.com](https://npcap.com) (check *"Install Npcap in WinPcap API-compatible Mode"*).

### 2. Environment Setup
```powershell
git clone https://github.com/SANJEEVIN2005/Intrusion_detection_system_cnn_lstm.git
cd Intrusion_detection_system_cnn_lstm
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Main Application
Open **PowerShell as Administrator** (required for packet sniffing and firewall rules):
```powershell
python app.py
```
* **Desktop SOC Dashboard:** Open [http://localhost:5000](http://localhost:5000)
* **Smartphone Remote:** Click **"Mobile Remote"** on the dashboard and scan the QR code with your phone!

---

## 🔬 Research & Benchmarking CLI Commands

* **Run Empirical Ablation Study:**
  ```powershell
  python src/ablation_study.py
  ```
* **Run Adversarial Robustness Benchmark (FGSM/PGD):**
  ```powershell
  python src/adversarial_eval.py
  ```
* **Generate Publication 300 DPI Figures:**
  ```powershell
  python src/publication_plots.py
  ```
* **Run All 42 Unit Tests:**
  ```powershell
  pytest
  ```

---

## 📜 License
This project is licensed under the MIT License.
