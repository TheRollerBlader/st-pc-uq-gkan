"""Real-time NOAA SWPC Solar Wind & Geomagnetic Ingestion Client.

Ingests L1 Lagrange point satellite observations (DSCOVR / ACE) from NOAA SWPC
REST feeds, aligns plasma and interplanetary magnetic field (IMF) streams, computes
the Newell empirical magnetopause coupling function, and constructs tensors for
the Graph Kolmogorov-Arnold Network (GKAN) GIC forecasting pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import urllib.request
import numpy as np


@dataclass
class SolarWindObservation:
    time_tag: str
    timestamp_epoch: float
    bx: float
    by: float
    bz: float
    bt: float
    speed: float
    density: float
    temperature: float
    newell_coupling: float
    is_anomaly: bool


class NoaaSwpcIngester:
    """Parses and aligns NOAA SWPC solar wind plasma, magnetic field, and planetary Kp feeds."""

    MAG_5M_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json"
    PLASMA_5M_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json"
    KP_1M_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

    def __init__(self, southward_bz_threshold: float = -10.0, high_speed_threshold: float = 600.0):
        self.southward_bz_threshold = southward_bz_threshold
        self.high_speed_threshold = high_speed_threshold

    @staticmethod
    def parse_utc_epoch(time_tag: str) -> Optional[float]:
        """Parses NOAA SWPC UTC timestamps ('YYYY-MM-DD HH:MM:SS(.fff)' or ISO-8601) to epoch seconds."""
        cleaned = time_tag.strip()
        if not cleaned:
            return None
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(cleaned)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    @staticmethod
    def compute_newell_coupling(speed: float, by: float, bz: float) -> float:
        """Computes Newell empirical magnetopause reconnection rate (Wb/s).

        Formula: dPhi/dt = v^(4/3) * (by^2 + bz^2)^(1/3) * sin^(8/3)(theta / 2)
        where theta = atan2(by, bz) is the IMF clock angle.
        """
        if speed <= 0:
            return 0.0
        bt = math.sqrt(by * by + bz * bz)
        if bt < 1e-4:
            return 0.0
        # Clock angle in radians
        theta = math.atan2(by, bz)
        sin_half = math.sin(theta / 2.0)
        clock_term = abs(sin_half) ** (8.0 / 3.0)
        coupling = (speed ** (4.0 / 3.0)) * (bt ** (2.0 / 3.0)) * clock_term
        return round(coupling, 3)

    def parse_mag_payload(self, raw_data: List[List[Any]]) -> Dict[str, Dict[str, float]]:
        """Parses NOAA SWPC mag-5-minute.json matrix.

        Header: ["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "lon_gsm", "lat_gsm", "bt"]
        """
        if not raw_data or len(raw_data) < 2:
            return {}
        header = raw_data[0]
        records: Dict[str, Dict[str, float]] = {}

        try:
            time_idx = header.index("time_tag")
            bx_idx = header.index("bx_gsm")
            by_idx = header.index("by_gsm")
            bz_idx = header.index("bz_gsm")
            bt_idx = header.index("bt")
        except ValueError:
            return {}

        for row in raw_data[1:]:
            try:
                t = str(row[time_idx])
                bx = float(row[bx_idx]) if row[bx_idx] is not None else 0.0
                by = float(row[by_idx]) if row[by_idx] is not None else 0.0
                bz = float(row[bz_idx]) if row[bz_idx] is not None else 0.0
                bt = float(row[bt_idx]) if row[bt_idx] is not None else math.sqrt(bx * bx + by * by + bz * bz)
                records[t] = {"bx": bx, "by": by, "bz": bz, "bt": bt}
            except (ValueError, IndexError):
                continue

        return records

    def parse_plasma_payload(self, raw_data: List[List[Any]]) -> Dict[str, Dict[str, float]]:
        """Parses NOAA SWPC plasma-5-minute.json matrix.

        Header: ["time_tag", "density", "speed", "temperature"]
        """
        if not raw_data or len(raw_data) < 2:
            return {}
        header = raw_data[0]
        records: Dict[str, Dict[str, float]] = {}

        try:
            time_idx = header.index("time_tag")
            dens_idx = header.index("density")
            speed_idx = header.index("speed")
            temp_idx = header.index("temperature")
        except ValueError:
            return {}

        for row in raw_data[1:]:
            try:
                t = str(row[time_idx])
                density = float(row[dens_idx]) if row[dens_idx] is not None else 0.0
                speed = float(row[speed_idx]) if row[speed_idx] is not None else 400.0
                temperature = float(row[temp_idx]) if row[temp_idx] is not None else 100000.0
                records[t] = {"density": density, "speed": speed, "temperature": temperature}
            except (ValueError, IndexError):
                continue

        return records

    def align_and_build_series(
        self,
        mag_records: Dict[str, Dict[str, float]],
        plasma_records: Dict[str, Dict[str, float]],
    ) -> List[SolarWindObservation]:
        """Aligns IMF magnetic fields and solar wind plasma on matching UTC time tags."""
        common_times = sorted(set(mag_records.keys()) & set(plasma_records.keys()))
        observations: List[SolarWindObservation] = []

        for t in common_times:
            mag = mag_records[t]
            plas = plasma_records[t]

            speed = plas["speed"]
            bx, by, bz, bt = mag["bx"], mag["by"], mag["bz"], mag["bt"]
            density, temp = plas["density"], plas["temperature"]

            coupling = self.compute_newell_coupling(speed, by, bz)
            is_anomaly = (bz <= self.southward_bz_threshold) or (speed >= self.high_speed_threshold)

            # Parse true UTC timestamp epoch when available, falling back to 5-min index
            epoch = self.parse_utc_epoch(t)
            obs = SolarWindObservation(
                time_tag=t,
                timestamp_epoch=epoch if epoch is not None else float(len(observations) * 300),
                bx=bx,
                by=by,
                bz=bz,
                bt=bt,
                speed=speed,
                density=density,
                temperature=temp,
                newell_coupling=coupling,
                is_anomaly=is_anomaly,
            )
            observations.append(obs)

        return observations

    def parse_kp_payload(self, raw_data: List[List[Any]]) -> Dict[str, float]:
        """Parses NOAA SWPC planetary K-index matrix into {time_tag: kp_value}."""
        if not raw_data or len(raw_data) < 2:
            return {}
        header = [str(col).lower() for col in raw_data[0]]
        try:
            time_idx = header.index("time_tag")
            kp_idx = header.index("kp") if "kp" in header else header.index("kp_index")
        except ValueError:
            return {}

        kp_map: Dict[str, float] = {}
        for row in raw_data[1:]:
            try:
                t = str(row[time_idx])
                val = float(row[kp_idx])
                if math.isfinite(val):
                    kp_map[t] = val
            except (ValueError, TypeError, IndexError):
                continue
        return kp_map

    @staticmethod
    def _default_fetch_json(url: str, timeout: float = 10.0) -> Any:
        req = urllib.request.Request(url, headers={"User-Agent": "ST-PC-UQ-GKAN/1.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def fetch_live_swpc_window(
        self,
        fetch_json: Optional[Callable[[str], Any]] = None,
        include_kp: bool = True,
    ) -> Dict[str, Any]:
        """Fetches and aligns live NOAA SWPC mag, plasma, and optional Kp feeds."""
        fetcher = fetch_json or self._default_fetch_json
        mag_raw = fetcher(self.MAG_5M_URL)
        plasma_raw = fetcher(self.PLASMA_5M_URL)
        mag_map = self.parse_mag_payload(mag_raw)
        plasma_map = self.parse_plasma_payload(plasma_raw)
        observations = self.align_and_build_series(mag_map, plasma_map)
        kp_series = self.parse_kp_payload(fetcher(self.KP_1M_URL)) if include_kp else {}

        return {
            "observations": observations,
            "solar_drivers": self.to_solar_driver_array(observations),
            "kp_series": kp_series,
            "anomaly_count": sum(1 for o in observations if o.is_anomaly),
        }

    def to_solar_driver_array(self, observations: List[SolarWindObservation]) -> np.ndarray:
        """Converts observations into [N, 5] array matching SOLAR_COLUMNS: (bx, by, bz, speed, density)."""
        if not observations:
            return np.empty((0, 5), dtype=np.float32)

        arr = np.zeros((len(observations), 5), dtype=np.float32)
        for i, o in enumerate(observations):
            arr[i, 0] = o.bx
            arr[i, 1] = o.by
            arr[i, 2] = o.bz
            arr[i, 3] = o.speed
            arr[i, 4] = o.density

        return arr
