"""
test_adversarial.py - Unit tests for FGSM and PGD adversarial attack functions.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from adversarial_eval import fgsm_attack, pgd_attack
from model import build_cnn_lstm_attention


def test_fgsm_attack_properties():
    model = build_cnn_lstm_attention(input_dim=80, num_classes=4)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    X = torch.rand(4, 1, 80)
    y = torch.tensor([0, 1, 2, 3])
    epsilon = 0.05

    X_adv = fgsm_attack(model, X, y, epsilon, criterion)

    # 1. Same shape
    assert X_adv.shape == X.shape
    # 2. Strict range [0, 1]
    assert (X_adv >= 0.0).all() and (X_adv <= 1.0).all()
    # 3. Maximum perturbation bounded by epsilon
    diff = torch.abs(X_adv - X)
    assert diff.max().item() <= epsilon + 1e-6


def test_pgd_attack_properties():
    model = build_cnn_lstm_attention(input_dim=80, num_classes=4)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    X = torch.rand(4, 1, 80)
    y = torch.tensor([0, 1, 2, 3])
    epsilon = 0.08

    X_pgd = pgd_attack(model, X, y, epsilon, num_iter=5, criterion=criterion)

    # 1. Same shape
    assert X_pgd.shape == X.shape
    # 2. Strict range [0, 1]
    assert (X_pgd >= 0.0).all() and (X_pgd <= 1.0).all()
    # 3. Maximum perturbation bounded by epsilon
    diff = torch.abs(X_pgd - X)
    assert diff.max().item() <= epsilon + 1e-6


def test_zero_epsilon_identity():
    model = build_cnn_lstm_attention(input_dim=80, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    X = torch.rand(2, 1, 80)
    y = torch.tensor([0, 1])

    X_adv = fgsm_attack(model, X, y, epsilon=0.0, criterion=criterion)
    assert torch.equal(X, X_adv)

    X_pgd = pgd_attack(model, X, y, epsilon=0.0, criterion=criterion)
    assert torch.equal(X, X_pgd)
