"""Forecast scores used by the demo and the tests."""

from __future__ import annotations

import numpy as np


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(error**2)))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(error)))


def pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    left = np.asarray(prediction, dtype=np.float64).reshape(-1)
    right = np.asarray(target, dtype=np.float64).reshape(-1)
    if left.std() < 1e-8 or right.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def nash_sutcliffe_efficiency(prediction: np.ndarray, target: np.ndarray) -> float:
    """Computes Nash-Sutcliffe Efficiency (1 - SSE / SST) for geomagnetic series."""
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    obs = np.asarray(target, dtype=np.float64).reshape(-1)
    denom = float(np.sum((obs - obs.mean()) ** 2))
    if denom <= 1e-12:
        return 0.0
    num = float(np.sum((obs - pred) ** 2))
    return float(1.0 - num / denom)


def interval_coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
) -> float:
    """Empirical coverage fraction of conformal prediction intervals [lower, upper]."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    obs = np.asarray(target, dtype=np.float64)
    covered = (obs >= lo) & (obs <= hi)
    return float(np.mean(covered))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean width of conformal prediction intervals (sharpness metric)."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    return float(np.mean(np.maximum(0.0, hi - lo)))


def winkler_interval_score(
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Proper scoring rule balancing interval sharpness and miscoverage penalty."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    obs = np.asarray(target, dtype=np.float64)
    width = np.maximum(0.0, hi - lo)
    below = (2.0 / alpha) * np.maximum(0.0, lo - obs)
    above = (2.0 / alpha) * np.maximum(0.0, obs - hi)
    return float(np.mean(width + below + above))
