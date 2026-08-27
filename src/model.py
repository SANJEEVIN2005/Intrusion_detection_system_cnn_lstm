"""
model.py

Deep Learning Architectures for IoT Intrusion Detection:
  1. CNNBaselineIDS: 1D-CNN Baseline Architecture (Khan et al., ICIS 2024 [1])
  2. CNNLSTMIDS: Hybrid CNN-LSTM Model (Ikhlas et al., arXiv:2606.05776)
  3. CNNLSTMAttentionIDS: Proposed Novel CNN-LSTM-Attention Architecture with Self-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention1D(nn.Module):
    """
    Temporal Feature Self-Attention Mechanism.
    Computes dynamic attention weights across the sequential feature representations
    extracted by the LSTM layer, learning to focus on critical attack indicators.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_outputs: torch.Tensor, return_attention: bool = False):
        # lstm_outputs: (batch_size, seq_len, hidden_dim)
        scores = self.projection(lstm_outputs)                     # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)                        # (batch_size, seq_len, 1)
        context = torch.sum(lstm_outputs * weights, dim=1)        # (batch_size, hidden_dim)

        if return_attention:
            return context, weights.squeeze(-1)                   # (batch_size, hidden_dim), (batch_size, seq_len)
        return context


class CNNLSTMAttentionIDS(nn.Module):
    """
    Novel CNN-LSTM-Attention Architecture for IoT Intrusion Detection.
    Combines spatial feature extraction (CNN), temporal sequence learning (LSTM),
    and adaptive focus mechanism (Self-Attention).
    """

    def __init__(self, input_dim: int, num_classes: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Spatial Feature Extraction Stage (1D-CNN)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Temporal Sequence Learning Stage (LSTM)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)

        # Adaptive Focus Stage (Self-Attention)
        self.attention = SelfAttention1D(hidden_dim=64)

        # Classification Head
        self.fc1 = nn.Linear(64, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (batch_size, 1, input_dim)
        x = self.pool1(self.relu1(self.conv1(x)))                 # (batch, 64, input_dim/2)
        x = self.pool2(self.relu2(self.conv2(x)))                 # (batch, 32, input_dim/4)

        # Prepare for LSTM (batch, seq_len, features)
        x = x.transpose(1, 2)                                      # (batch, input_dim/4, 32)
        lstm_out, _ = self.lstm(x)                                 # (batch, input_dim/4, 64)

        # Apply Self-Attention
        if return_attention:
            context, attn_weights = self.attention(lstm_out, return_attention=True)
        else:
            context = self.attention(lstm_out, return_attention=False)
            attn_weights = None

        # Classification Head
        out = self.dropout(self.relu3(self.fc1(context)))
        logits = self.fc2(out)

        if return_attention:
            return logits, attn_weights
        return logits


class CNNLSTMIDS(nn.Module):
    """
    Original CNN-LSTM model from Ikhlas et al. (arXiv:2606.05776).
    Uses the final hidden state of LSTM without attention.
    """

    def __init__(self, input_dim: int, num_classes: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)

        self.fc1 = nn.Linear(64, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        x = h_n.squeeze(0)
        x = self.dropout(self.relu3(self.fc1(x)))
        logits = self.fc2(x)
        return logits


class CNNBaselineIDS(nn.Module):
    """
    Baseline 1D-CNN model (Khan et al. [1]).
    Uses pure Flatten + Dense layers without sequential modeling.
    """

    def __init__(self, input_dim: int, num_classes: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        flatten_dim = 32 * (input_dim // 4)
        self.fc1 = nn.Linear(flatten_dim, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.dropout(self.relu3(self.fc1(x)))
        logits = self.fc2(x)
        return logits


def build_cnn_lstm(input_dim: int, num_classes: int = 4, seed: int = 42) -> CNNLSTMIDS:
    """Builds standard CNN-LSTM model."""
    torch.manual_seed(seed)
    return CNNLSTMIDS(input_dim=input_dim, num_classes=num_classes)


def build_cnn_lstm_attention(input_dim: int, num_classes: int = 4, seed: int = 42) -> CNNLSTMAttentionIDS:
    """Builds proposed CNN-LSTM-Attention model."""
    torch.manual_seed(seed)
    return CNNLSTMAttentionIDS(input_dim=input_dim, num_classes=num_classes)


def build_cnn_baseline(input_dim: int, num_classes: int = 4, seed: int = 42) -> CNNBaselineIDS:
    """Builds baseline 1D-CNN model."""
    torch.manual_seed(seed)
    return CNNBaselineIDS(input_dim=input_dim, num_classes=num_classes)
