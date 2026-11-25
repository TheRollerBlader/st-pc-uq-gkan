"""SVG visualization generator for conformal geomagnetic forecast intervals."""

from __future__ import annotations

from html import escape
from typing import Sequence

import numpy as np

from stpcuq_gkan.metrics import interval_coverage, nash_sutcliffe_efficiency


def render_conformal_forecast_svg(
    prediction: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray | None = None,
    hazard_threshold: float | None = None,
    width: int = 760,
    height: int = 320,
    title: str = "Geomagnetic Forecast & Conformal Interval",
) -> str:
    """Renders a standalone SVG chart of point forecasts, conformal bounds, and observations."""
    pred_arr = np.asarray(prediction, dtype=np.float64).reshape(-1)
    lo_arr = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi_arr = np.asarray(upper, dtype=np.float64).reshape(-1)

    if not (pred_arr.size == lo_arr.size == hi_arr.size):
        raise ValueError(
            "prediction, lower, and upper must have identical lengths; "
            f"got {pred_arr.size}, {lo_arr.size}, {hi_arr.size}"
        )
    if pred_arr.size == 0:
        raise ValueError("prediction series must be non-empty")

    obs_arr: np.ndarray | None = None
    if target is not None:
        obs_arr = np.asarray(target, dtype=np.float64).reshape(-1)
        if obs_arr.size != pred_arr.size:
            raise ValueError(
                f"target length {obs_arr.size} does not match prediction length {pred_arr.size}"
            )

    pad_left, pad_right, pad_top, pad_bottom = 56, 20, 38, 36
    inner_w = max(1, width - pad_left - pad_right)
    inner_h = max(1, height - pad_top - pad_bottom)

    stacked = [pred_arr, lo_arr, hi_arr]
    if obs_arr is not None:
        stacked.append(obs_arr)
    combined = np.concatenate(stacked)
    if hazard_threshold is not None and np.isfinite(hazard_threshold):
        combined = np.append(combined, float(hazard_threshold))

    finite_vals = combined[np.isfinite(combined)]
    y_min = float(np.min(finite_vals)) if finite_vals.size > 0 else 0.0
    y_max = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
    if abs(y_max - y_min) < 1e-6:
        y_min -= 1.0
        y_max += 1.0
    margin = 0.08 * (y_max - y_min)
    y_min -= margin
    y_max += margin

    n = pred_arr.size

    def x_at(idx: int) -> float:
        if n == 1:
            return pad_left + inner_w / 2.0
        return pad_left + (idx / (n - 1)) * inner_w

    def y_at(val: float) -> float:
        safe = val if np.isfinite(val) else y_min
        return pad_top + (1.0 - (safe - y_min) / (y_max - y_min)) * inner_h

    upper_pts = [(round(x_at(i), 2), round(y_at(float(hi_arr[i])), 2)) for i in range(n)]
    lower_pts = [(round(x_at(i), 2), round(y_at(float(lo_arr[i])), 2)) for i in range(n - 1, -1, -1)]
    band_points = " ".join(f"{px},{py}" for px, py in upper_pts + lower_pts)

    pred_points = " ".join(
        f"{round(x_at(i), 2)},{round(y_at(float(pred_arr[i])), 2)}" for i in range(n)
    )

    subtitle = f"n={n} steps"
    if obs_arr is not None:
        cov = interval_coverage(lo_arr, hi_arr, obs_arr) * 100.0
        nse = nash_sutcliffe_efficiency(pred_arr, obs_arr)
        subtitle = f"Coverage {cov:.1f}% · NSE {nse:.3f}"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{escape(title)}">',
        f'  <rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="#0b1120" />',
        f'  <text x="{pad_left}" y="20" fill="#e2e8f0" font-family="sans-serif" '
        f'font-size="13" font-weight="600">{escape(title)}</text>',
        f'  <text x="{width - pad_right}" y="20" text-anchor="end" fill="#38bdf8" '
        f'font-family="monospace" font-size="11">{escape(subtitle)}</text>',
        f'  <rect x="{pad_left}" y="{pad_top}" width="{inner_w}" height="{inner_h}" '
        f'fill="#111827" stroke="#334155" stroke-width="1" />',
        f'  <polygon data-layer="conformal-band" points="{band_points}" '
        f'fill="#38bdf8" fill-opacity="0.22" />',
        f'  <polyline data-layer="prediction" fill="none" stroke="#38bdf8" '
        f'stroke-width="2.2" points="{pred_points}" />',
    ]

    if obs_arr is not None:
        obs_points = " ".join(
            f"{round(x_at(i), 2)},{round(y_at(float(obs_arr[i])), 2)}" for i in range(n)
        )
        lines.append(
            f'  <polyline data-layer="target" fill="none" stroke="#fbbf24" '
            f'stroke-width="1.8" stroke-dasharray="4 3" points="{obs_points}" />'
        )

    if hazard_threshold is not None and np.isfinite(hazard_threshold):
        hy = round(y_at(float(hazard_threshold)), 2)
        lines.append(
            f'  <line data-layer="hazard-threshold" x1="{pad_left}" y1="{hy}" '
            f'x2="{pad_left + inner_w}" y2="{hy}" stroke="#f87171" '
            f'stroke-width="1.5" stroke-dasharray="6 4" />'
        )

    lines.append(
        f'  <text x="{pad_left - 6}" y="{pad_top + 10}" text-anchor="end" '
        f'fill="#94a3b8" font-family="monospace" font-size="10">{y_max:.1f}</text>'
    )
    lines.append(
        f'  <text x="{pad_left - 6}" y="{pad_top + inner_h}" text-anchor="end" '
        f'fill="#94a3b8" font-family="monospace" font-size="10">{y_min:.1f}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)
