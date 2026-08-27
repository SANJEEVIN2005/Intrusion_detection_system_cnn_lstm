"""
test_xai_and_notifier.py - Unit tests for XAI Explainer and Threat Notifier.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xai_explainer import XAIExplainer
from notifier import ThreatNotifier


@pytest.fixture
def artifacts_dir():
    return os.path.join(os.path.dirname(__file__), "..", "artifacts")


def test_xai_explainer_attribution(artifacts_dir):
    explainer = XAIExplainer(artifacts_dir=artifacts_dir)
    test_vec = np.random.randn(80)
    raw_dict = {"syn_flag_cnt": 100, "flow_pkts_s": 2500, "flow_byts_s": 100000}

    res_dos = explainer.explain(raw_dict, test_vec, "DoS", 99.2)
    assert "top_features" in res_dos
    assert len(res_dos["top_features"]) > 0
    assert "insights" in res_dos
    assert len(res_dos["insights"]) > 0
    assert any("SYN" in str(feat.get("name", "")) for feat in res_dos["top_features"])

    res_benign = explainer.explain(raw_dict, test_vec, "Benign", 98.0)
    assert len(res_benign["insights"]) > 0


def test_threat_notifier_config(tmp_path):
    cfg_file = str(tmp_path / "alerts_config.json")
    notifier = ThreatNotifier(config_path=cfg_file)

    saved = notifier.save_config({
        "telegram_enabled": True,
        "telegram_bot_token": "test_token_123",
        "telegram_chat_id": "test_chat_456"
    })
    assert saved["telegram_enabled"] is True
    assert saved["telegram_bot_token"] == "test_token_123"

    loaded = notifier.get_config()
    assert loaded["telegram_chat_id"] == "test_chat_456"


def test_threat_notifier_test_ping(tmp_path):
    cfg_file = str(tmp_path / "alerts_config.json")
    notifier = ThreatNotifier(config_path=cfg_file)
    res = notifier.send_test_alert()
    assert "warning" in res or "telegram" in res or "discord" in res
