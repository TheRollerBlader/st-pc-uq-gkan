"""Uncertainty quantification calibration and physical Newell coupling verification tests.
Scheduled Milestone: Testing & Verification (2026-11-13).

Validates:
1. Adaptive conformal inference empirical coverage convergence (1 - alpha_target).
2. Newell coupling southward IMF enhancement and tensor/numpy consistency.
3. GIC risk classification under synthetic geomagnetic storm perturbations.
"""

import numpy as np
import pytest
import torch

from stpcuq_gkan.conformal import AdaptiveConformalInference
from stpcuq_gkan.physics import (
    newell_coupling,
    newell_coupling_numpy,
    gic_risk_level,
    gic_risk_level_numpy,
    RISK_MODERATE_VKM,
    RISK_SEVERE_VKM,
)


class TestAdaptiveConformalCalibration:
    def test_empirical_coverage_converges_near_target(self):
        """On a stationary residual distribution, online adaptive conformal inference
        coverage must converge near (1 - alpha_target) = 0.90 within reasonable empirical margin.
        """
        np.random.seed(42)
        target_alpha = 0.10
        aci = AdaptiveConformalInference(alpha_target=target_alpha, gamma=0.02, window=256)

        # Generate 1000 standard half-normal residuals
        synthetic_residuals = np.abs(np.random.normal(loc=0.0, scale=2.5, size=1000))
        result = aci.calibrate(synthetic_residuals, fallback=5.0)

        # Coverage should be within [0.85, 0.95] for target 0.90
        assert 0.84 <= result["coverage"] <= 0.96, f"Expected ~0.90 coverage, got {result['coverage']}"
        assert result["mean_half_width"] > 0.0

    def test_interval_widens_under_storm_burst_miscoverages(self):
        """A burst of large storm residuals must trigger miscoverage errors,
        lowering alpha and widening subsequent half-width intervals.
        """
        aci = AdaptiveConformalInference(alpha_target=0.10, gamma=0.05, window=64)

        # Warm up on quiet baseline
        for _ in range(50):
            aci.update(0.5, fallback=2.0)
        quiet_width = aci.half_width()

        # Simulate sudden geomagnetic storm shock (extreme residuals)
        for _ in range(10):
            aci.update(15.0, fallback=2.0)
        storm_width = aci.half_width()

        assert storm_width > quiet_width, "Storm shocks must expand uncertainty intervals"


class TestNewellCouplingPhysics:
    def test_southward_imf_generates_higher_coupling_than_northward(self):
        """Dayside magnetic reconnection is maximized when IMF Bz is strongly negative (southward)."""
        speed = 500.0  # km/s
        by = 2.0       # nT
        bz_north = 15.0  # Due North (quiet)
        bz_south = -15.0 # Due South (geoeffective)

        coupling_north = newell_coupling_numpy(
            np.array([speed]), np.array([by]), np.array([bz_north])
        )[0]
        coupling_south = newell_coupling_numpy(
            np.array([speed]), np.array([by]), np.array([bz_south])
        )[0]

        assert coupling_south > coupling_north * 5.0, (
            f"Southward IMF coupling ({coupling_south:.1f}) should dominate northward ({coupling_north:.1f})"
        )

    def test_torch_numpy_coupling_numerical_equivalence(self):
        """PyTorch tensor and NumPy implementations of Newell coupling must match to high precision."""
        speeds = np.array([350.0, 480.0, 720.0], dtype=np.float32)
        bys = np.array([-4.5, 2.1, 8.0], dtype=np.float32)
        bzs = np.array([-12.0, 3.5, -22.0], dtype=np.float32)

        res_numpy = newell_coupling_numpy(speeds, bys, bzs)
        res_torch = newell_coupling(
            torch.from_numpy(speeds),
            torch.from_numpy(bys),
            torch.from_numpy(bzs),
        ).numpy()

        np.testing.assert_allclose(res_numpy, res_torch, rtol=1e-4, atol=1e-5)


class TestGICRiskClassification:
    def test_gic_risk_threshold_boundaries(self):
        """Evaluates geoelectric field magnitude discretization into quiet (0), moderate (1), and severe (2)."""
        fields = np.array([0.2, RISK_MODERATE_VKM - 0.01, RISK_MODERATE_VKM, 3.2, RISK_SEVERE_VKM, RISK_SEVERE_VKM + 1.5])
        expected = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)

        labels_numpy = gic_risk_level_numpy(fields)
        labels_torch = gic_risk_level(torch.from_numpy(fields)).numpy()

        np.testing.assert_array_equal(labels_numpy, expected)
        np.testing.assert_array_equal(labels_torch, expected)
