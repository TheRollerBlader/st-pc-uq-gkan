"""ST-PC-UQ-GKAN reference implementation.

Spatiotemporal, physics-constrained, uncertainty-quantified graph
Kolmogorov-Arnold network for localized geomagnetic and GIC-risk forecasts.
"""

from stpcuq_gkan.model import ModelConfig, STPCUQGKAN

__all__ = ["ModelConfig", "STPCUQGKAN"]
__version__ = "0.1.0"
