"""Solar-wind coupling and GIC risk thresholds.

The Newell coupling function is the differentiable physics residual used by
subsystem 4. It estimates the rate of magnetic flux opening at the dayside
magnetopause:

    dΦ_MP/dt = v_sw^(4/3) * B_T^(2/3) * sin(θ_c / 2)^(8/3)

with B_T = sqrt(B_y^2 + B_z^2) and θ_c the IMF clock angle.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# Geoelectric magnitude thresholds from the architecture note, in V/km.
RISK_MODERATE_VKM = 1.0
RISK_SEVERE_VKM = 5.0
RISK_NAMES = ("quiet", "moderate", "severe")


def newell_coupling(
    speed: Tensor,
    by: Tensor,
    bz: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Newell dayside coupling. ``speed`` is km/s and the IMF components are nT."""
    transverse = torch.sqrt(by.square() + bz.square() + eps)
    clock = torch.acos(torch.clamp(bz / transverse, -1.0, 1.0))
    speed_term = torch.clamp(speed, min=0).pow(4.0 / 3.0)
    field_term = transverse.pow(2.0 / 3.0)
    clock_term = torch.sin(clock / 2.0).clamp(min=0).pow(8.0 / 3.0)
    return speed_term * field_term * clock_term


def newell_coupling_numpy(
    speed: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    transverse = np.sqrt(by**2 + bz**2 + eps)
    clock = np.arccos(np.clip(bz / transverse, -1.0, 1.0))
    return (
        np.clip(speed, 0, None) ** (4.0 / 3.0)
        * transverse ** (2.0 / 3.0)
        * np.sin(clock / 2.0).clip(min=0) ** (8.0 / 3.0)
    )


def gic_risk_level(e_magnitude: Tensor) -> Tensor:
    """Map |E| in V/km onto quiet / moderate / severe integer labels."""
    moderate = e_magnitude >= RISK_MODERATE_VKM
    severe = e_magnitude > RISK_SEVERE_VKM
    return moderate.long() + severe.long()


def gic_risk_level_numpy(e_magnitude: np.ndarray) -> np.ndarray:
    labels = np.zeros(e_magnitude.shape, dtype=np.int64)
    labels[e_magnitude >= RISK_MODERATE_VKM] = 1
    labels[e_magnitude > RISK_SEVERE_VKM] = 2
    return labels
