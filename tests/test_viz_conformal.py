"""Unit tests for the conformal geomagnetic forecast SVG visualizer."""

import numpy as np
import pytest

from stpcuq_gkan.viz_conformal import render_conformal_forecast_svg


def test_renders_conformal_band_prediction_and_metrics_subtitle():
    pred = np.array([10.0, 20.0, 35.0, 25.0])
    lower = pred - 5.0
    upper = pred + 5.0
    target = np.array([11.0, 19.0, 34.0, 26.0])

    svg = render_conformal_forecast_svg(
        prediction=pred,
        lower=lower,
        upper=upper,
        target=target,
        hazard_threshold=30.0,
        title="GIC Storm Horizon",
    )

    assert svg.startswith("<svg ")
    assert svg.endswith("</svg>")
    assert "GIC Storm Horizon" in svg
    assert "Coverage 100.0%" in svg
    assert 'data-layer="conformal-band"' in svg
    assert 'data-layer="prediction"' in svg
    assert 'data-layer="target"' in svg
    assert 'data-layer="hazard-threshold"' in svg


def test_raises_on_mismatched_lengths_or_empty_series():
    with pytest.raises(ValueError, match="identical lengths"):
        render_conformal_forecast_svg([1.0, 2.0], [0.5], [1.5, 2.5])

    with pytest.raises(ValueError, match="non-empty"):
        render_conformal_forecast_svg([], [], [])

    with pytest.raises(ValueError, match="target length"):
        render_conformal_forecast_svg([1.0, 2.0], [0.5, 1.5], [1.5, 2.5], target=[1.0])
