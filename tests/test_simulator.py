import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attack_simulator import AttackSimulator


@pytest.fixture
def simulator():
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    return AttackSimulator(artifacts_dir=artifacts_dir)


def test_simulate_dos(simulator):
    result = simulator.generate_dos_flow()
    assert "predicted_class" in result
    assert "confidence" in result
    assert result["protocol"] == "TCP"
    assert result["metrics"]["total_pkts"] >= 50
    assert result["metrics"]["syn_flags"] > 0


def test_simulate_ddos(simulator):
    result = simulator.generate_ddos_flow()
    assert "predicted_class" in result
    assert "confidence" in result
    assert result["protocol"] in ("TCP", "UDP")
    assert result["metrics"]["total_pkts"] >= 100


def test_simulate_recon(simulator):
    result = simulator.generate_recon_flow()
    assert "predicted_class" in result
    assert "confidence" in result
    assert result["protocol"] == "TCP"
    assert result["metrics"]["total_pkts"] <= 5
