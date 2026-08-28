"""
test_quantization.py - Unit tests for PyTorch INT8 Edge Quantization.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quantize_model import quantize_network, measure_model_size_kb
from model import build_cnn_lstm_attention, build_cnn_lstm, build_cnn_baseline


def test_quantization_forward_pass():
    model_fp32 = build_cnn_lstm_attention(input_dim=80, num_classes=4)
    model_int8 = quantize_network(model_fp32)

    dummy_input = torch.rand(4, 1, 80)
    with torch.no_grad():
        out_fp32 = model_fp32(dummy_input)
        out_int8 = model_int8(dummy_input)

    # 1. Output shape must match exactly (batch_size, num_classes)
    assert out_int8.shape == (4, 4)

    # 2. Output cosine similarity fidelity > 0.98
    cos_sim = torch.nn.functional.cosine_similarity(out_fp32, out_int8).mean().item()
    assert cos_sim > 0.98


def test_quantization_size_reduction():
    model_fp32 = build_cnn_lstm_attention(input_dim=80, num_classes=4)
    model_int8 = quantize_network(model_fp32)

    size_fp32 = measure_model_size_kb(model_fp32)
    size_int8 = measure_model_size_kb(model_int8)

    # INT8 model must be substantially smaller than FP32
    assert size_int8 < size_fp32
    reduction_pct = (1.0 - (size_int8 / size_fp32)) * 100
    assert reduction_pct > 30.0


def test_quantize_all_architectures():
    models = [
        build_cnn_baseline(input_dim=80, num_classes=4),
        build_cnn_lstm(input_dim=80, num_classes=4),
        build_cnn_lstm_attention(input_dim=80, num_classes=4),
    ]

    dummy_input = torch.rand(2, 1, 80)
    for m in models:
        q_m = quantize_network(m)
        with torch.no_grad():
            out = q_m(dummy_input)
        assert out.shape == (2, 4)
