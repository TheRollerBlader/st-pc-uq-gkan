"""Unit tests for forecast skill and conformal uncertainty calibration metrics."""

import numpy as np
import pytest

from stpcuq_gkan.metrics import (
    interval_coverage,
    mean_interval_width,
    nash_sutcliffe_efficiency,
    winkler_interval_score,
)


def test_nash_sutcliffe_efficiency_perfect_and_mean_baseline():
    obs = np.array([1.0, 2.0, 4.0, 3.0])
    assert nash_sutcliffe_efficiency(obs, obs) == pytest.approx(1.0)
    mean_pred = np.full_like(obs, obs.mean())
    assert nash_sutcliffe_efficiency(mean_pred, obs) == pytest.approx(0.0)


def test_conformal_interval_coverage_width_and_winkler_score():
    target = np.array([0.5, 1.5, 2.5, 4.0])
    lower = np.array([0.0, 1.0, 2.0, 2.5])
    upper = np.array([1.0, 2.0, 3.0, 3.5])

    assert interval_coverage(lower, upper, target) == pytest.approx(0.75)
    assert mean_interval_width(lower, upper) == pytest.approx(1.0)

    # First 3 samples covered (score=1.0 each); 4th misses by 0.5 with 2/0.1=20x penalty (+10.0)
    score = winkler_interval_score(lower, upper, target, alpha=0.1)
    assert score == pytest.approx((1.0 + 1.0 + 1.0 + 11.0) / 4.0)


def test_winkler_interval_score_validates_alpha():
    with pytest.raises(ValueError):
        winkler_interval_score(np.array([0.0]), np.array([1.0]), np.array([0.5]), alpha=1.2)
