"""Unit tests for Newell solar-wind physics, GIC thresholds, and ReLoBRaLo loss balancing."""

import numpy as np
import pytest
import torch

from stpcuq_gkan.losses import LOSS_NAMES, ReLoBRaLo, task_losses
from stpcuq_gkan.physics import (
    gic_risk_level,
    gic_risk_level_numpy,
    newell_coupling,
    newell_coupling_numpy,
)


def test_newell_coupling_torch_and_numpy_parity_and_clamping():
    speed_np = np.array([-50.0, 0.0, 400.0, 750.0], dtype=np.float32)
    by_np = np.array([2.0, 0.0, -5.0, 3.0], dtype=np.float32)
    bz_np = np.array([-3.0, 0.0, 8.0, -12.0], dtype=np.float32)

    np_val = newell_coupling_numpy(speed_np, by_np, bz_np)
    torch_val = newell_coupling(
        torch.from_numpy(speed_np),
        torch.from_numpy(by_np),
        torch.from_numpy(bz_np),
    )

    assert np_val[0] == 0.0  # negative speed clamped to 0
    assert np_val[1] == 0.0  # zero speed yields zero coupling
    assert np.allclose(np_val, torch_val.numpy(), rtol=1e-4, atol=1e-5)


def test_gic_risk_level_torch_and_numpy_boundary_thresholds():
    values = np.array([0.0, 0.999, 1.0, 4.999, 5.0, 5.001], dtype=np.float32)
    expected = [0, 0, 1, 1, 1, 2]

    assert gic_risk_level_numpy(values).tolist() == expected
    assert gic_risk_level(torch.from_numpy(values)).tolist() == expected


def test_task_losses_handles_single_step_and_zero_driver_without_nans():
    torch.manual_seed(42)
    # Single timestep (T=1): d_sme is empty, physics loss must return 0 safely
    prediction = {
        "db": torch.randn(1, 3, 2, requires_grad=True),
        "sme": torch.randn(1, 1, requires_grad=True),
        "e_field": torch.randn(1, 3, requires_grad=True),
        "risk_logits": torch.randn(1, 3, 3, requires_grad=True),
    }
    batch = {
        "db": torch.zeros(1, 3, 2),
        "sme": torch.zeros(1, 1),
        "e_field": torch.zeros(1, 3),
        "risk": torch.zeros(1, 3, dtype=torch.long),
        "newell": torch.zeros(1),
    }
    losses = task_losses(prediction, batch, coupling_scale=torch.tensor(0.5))
    assert set(losses) == set(LOSS_NAMES)
    assert losses["physics"].item() == pytest.approx(0.0)
    assert all(torch.isfinite(val) for val in losses.values())


def test_relobralo_lookback_and_gradient_flow():
    torch.manual_seed(7)
    balancer = ReLoBRaLo(mix=0.5, temperature=1.0, lookback_probability=1.0)
    param = torch.tensor(2.0, requires_grad=True)

    for step in range(3):
        losses = {
            name: (param - float(idx + step)) ** 2 + 0.1
            for idx, name in enumerate(LOSS_NAMES)
        }
        total = balancer.combine(losses)
        assert torch.isfinite(total)
        assert torch.allclose(balancer.lambdas.sum(), torch.tensor(1.0), atol=1e-5)

    assert len(balancer.history) == 3
    total.backward()
    assert param.grad is not None and torch.isfinite(param.grad)
