"""
test_attention_model.py - Unit tests for CNN-LSTM-Attention and SelfAttention1D modules.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SelfAttention1D, CNNLSTMAttentionIDS, CNNBaselineIDS, CNNLSTMIDS, build_cnn_lstm_attention


def test_self_attention_module():
    batch_size = 4
    seq_len = 20
    hidden_dim = 64

    attn = SelfAttention1D(hidden_dim=hidden_dim)
    dummy_input = torch.randn(batch_size, seq_len, hidden_dim)

    # 1. Output shape check
    context = attn(dummy_input)
    assert context.shape == (batch_size, hidden_dim)

    # 2. Attention weights check
    context_w, weights = attn(dummy_input, return_attention=True)
    assert weights.shape == (batch_size, seq_len)
    # Weights across sequence must sum to 1.0
    sums = weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)


def test_cnn_lstm_attention_forward():
    batch_size = 8
    input_dim = 80
    num_classes = 4

    model = build_cnn_lstm_attention(input_dim=input_dim, num_classes=num_classes)
    dummy_input = torch.randn(batch_size, 1, input_dim)

    # Standard forward
    logits = model(dummy_input)
    assert logits.shape == (batch_size, num_classes)

    # Forward with attention weights
    logits_w, attn_weights = model(dummy_input, return_attention=True)
    assert logits_w.shape == (batch_size, num_classes)
    assert attn_weights.shape == (batch_size, input_dim // 4)


def test_cnn_lstm_attention_gradient_flow():
    model = build_cnn_lstm_attention(input_dim=80, num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    dummy_input = torch.randn(4, 1, 80)
    dummy_targets = torch.tensor([0, 1, 2, 3])

    optimizer.zero_grad()
    logits = model(dummy_input)
    loss = criterion(logits, dummy_targets)
    loss.backward()
    optimizer.step()

    # Verify gradients computed for attention layer
    for name, param in model.attention.named_parameters():
        assert param.grad is not None


def test_model_parameter_count():
    model_attn = build_cnn_lstm_attention(input_dim=80, num_classes=4)
    model_orig = CNNLSTMIDS(input_dim=80, num_classes=4)
    model_base = CNNBaselineIDS(input_dim=80, num_classes=4)

    params_attn = sum(p.numel() for p in model_attn.parameters() if p.requires_grad)
    params_orig = sum(p.numel() for p in model_orig.parameters() if p.requires_grad)
    params_base = sum(p.numel() for p in model_base.parameters() if p.requires_grad)

    print(f"CNN-LSTM-Attention params: {params_attn}")
    print(f"CNN-LSTM params: {params_orig}")
    print(f"CNN Baseline params: {params_base}")

    assert params_attn > 0
    assert params_attn < 100000  # Ultra-lightweight for IoT edge
