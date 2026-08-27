import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import build_cnn_lstm


def test_model_builds_without_error():
    model = build_cnn_lstm(input_dim=20, num_classes=4)
    assert model is not None


def test_model_output_shape_matches_batch():
    model = build_cnn_lstm(input_dim=20, num_classes=4)
    model.eval()
    dummy_X = torch.rand(8, 1, 20)  # (batch, channels, features)
    with torch.no_grad():
        preds = model(dummy_X)
    assert preds.shape == (8, 4)


def test_model_trains_one_epoch_smoke_test():
    model = build_cnn_lstm(input_dim=20, num_classes=4)
    model.train()
    dummy_X = torch.rand(32, 1, 20)
    dummy_y = torch.randint(0, 4, (32,))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    logits = model(dummy_X)
    loss = criterion(logits, dummy_y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).item()
