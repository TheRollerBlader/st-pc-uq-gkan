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
