"""Strict type-safe geomagnetic storm prediction contract for st-pc-uq-gkan.
Scheduled Milestone: Type Safety, Linting & Polish (2026-11-28).
"""

from typing import Tuple
from pydantic import BaseModel, Field


class GeomagneticHazardContract(BaseModel):
    """Validates ground magnetometer station delta-B vector bounds."""
    milestone_date: str = Field(default="2026-11-28")
    station_id: str = Field(..., min_length=3, max_length=6)
    delta_b_nt_per_min: float = Field(..., description="Rate of magnetic field perturbation")
    uncertainty_conformal_interval: Tuple[float, float] = Field(...)
    is_gic_alert: bool = Field(default=False)
