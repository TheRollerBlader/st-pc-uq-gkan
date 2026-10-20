"""Adaptive conformal inference for non-stationary storm residuals.

Gibbs and Candès (2021) update the miscoverage level online:

    α_{t+1} = α_t + γ (α_target - err_t)

``err_t`` is 1 when the observation falls outside the interval. A miss lowers
α, which raises the residual quantile and widens the next interval. The
architecture note writes the opposite sign; the prose there still asks for a
wider interval at storm onset, which is what this update does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AdaptiveConformalInference:
    alpha_target: float = 0.1
    gamma: float = 0.05
    window: int = 128
    alpha: float = 0.1
    scores: list[float] = field(default_factory=list)

    def half_width(self, fallback: float = 1.0) -> float:
        if len(self.scores) < 8:
            return fallback
        level = float(np.clip(1.0 - self.alpha, 0.5, 0.999))
        recent = self.scores[-self.window :]
        return float(np.quantile(recent, level))

    def update(self, absolute_residual: float, fallback: float = 1.0) -> bool:
        missed = absolute_residual > self.half_width(fallback)
        error = 1.0 if missed else 0.0
        self.alpha = float(
            np.clip(self.alpha + self.gamma * (self.alpha_target - error), 1e-3, 0.5)
        )
        self.scores.append(float(absolute_residual))
        return missed

    def calibrate(self, residuals: np.ndarray, fallback: float) -> dict[str, float]:
        misses = 0
        widths: list[float] = []
        for residual in np.abs(residuals.reshape(-1)):
            widths.append(self.half_width(fallback))
            misses += int(self.update(float(residual), fallback))
        count = max(len(widths), 1)
        covered = 1.0 - misses / count
        return {
            "coverage": covered,
            "mean_half_width": float(np.mean(widths)) if widths else fallback,
            "final_alpha": self.alpha,
        }
