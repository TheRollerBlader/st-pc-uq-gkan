"""Static geographic graph of ground magnetometer stations.

The architecture note keeps the adjacency fixed, because rebuilding a
magnetic-local-time graph every minute is expensive, and puts the rotating
coordinate system into the node features instead. Edge preference follows the
auroral electrojets: stations that share a geomagnetic latitude stay more
strongly connected than stations that are only close in longitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Approximate public coordinates for a 16-station demo graph.
# These are not an official SuperMAG extract.
STATIONS: tuple[tuple[str, str, float, float], ...] = (
    ("OTT", "Ottawa", 45.40, -75.55),
    ("FRD", "Fredericksburg", 38.20, -77.37),
    ("BOU", "Boulder", 40.14, -105.24),
    ("NEW", "Newport", 48.27, -117.12),
    ("CMO", "College", 64.87, -147.86),
    ("BLC", "Baker Lake", 64.32, -96.02),
    ("FCC", "Fort Churchill", 58.76, -94.09),
    ("YKC", "Yellowknife", 62.48, -114.48),
    ("MEA", "Meanook", 54.62, -113.35),
    ("VIC", "Victoria", 48.52, -123.42),
    ("SJG", "San Juan", 18.11, -66.15),
    ("HON", "Honolulu", 21.32, -158.00),
    ("GUA", "Guam", 13.59, 144.87),
    ("ABK", "Abisko", 68.36, 18.82),
    ("LER", "Lerwick", 60.14, -1.18),
    ("HER", "Hermanus", -34.43, 19.23),
)


@dataclass(frozen=True)
class StationGraph:
    codes: tuple[str, ...]
    names: tuple[str, ...]
    latitude: np.ndarray
    longitude: np.ndarray
    mlat: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray


def dipole_magnetic_latitude(
    latitude: np.ndarray,
    longitude: np.ndarray,
    pole_lat: float = 80.7,
    pole_lon: float = -72.6,
) -> np.ndarray:
    """Centered-dipole geomagnetic latitude. This is not an IGRF coordinate."""
    lat = np.radians(latitude)
    lon = np.radians(longitude)
    plat = np.radians(pole_lat)
    plon = np.radians(pole_lon)
    sin_mlat = np.sin(lat) * np.sin(plat) + np.cos(lat) * np.cos(plat) * np.cos(lon - plon)
    return np.degrees(np.arcsin(np.clip(sin_mlat, -1.0, 1.0)))


def _haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    radius = 6371.0
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_station_graph(k: int = 4) -> StationGraph:
    """k-nearest stations, scored so shared geomagnetic latitude is preferred."""
    codes = tuple(row[0] for row in STATIONS)
    names = tuple(row[1] for row in STATIONS)
    latitude = np.array([row[2] for row in STATIONS], dtype=np.float64)
    longitude = np.array([row[3] for row in STATIONS], dtype=np.float64)
    mlat = dipole_magnetic_latitude(latitude, longitude)
    distance = _haversine_km(latitude, longitude)
    dmlat = np.abs(mlat[:, None] - mlat[None, :])
    score = distance / 800.0 + 8.0 * dmlat
    np.fill_diagonal(score, np.inf)

    pairs: set[tuple[int, int]] = set()
    for node in range(len(STATIONS)):
        neighbors = np.argsort(score[node])[:k]
        for other in neighbors:
            a, b = (node, int(other)) if node < other else (int(other), node)
            pairs.add((a, b))

    sources: list[int] = []
    targets: list[int] = []
    attrs: list[list[float]] = []
    for left, right in sorted(pairs):
        for src, dst in ((left, right), (right, left)):
            sources.append(src)
            targets.append(dst)
            attrs.append(
                [
                    float(distance[src, dst] / 5000.0),
                    float(dmlat[src, dst] / 30.0),
                    float(np.exp(-dmlat[src, dst] / 8.0)),
                ]
            )

    return StationGraph(
        codes=codes,
        names=names,
        latitude=latitude,
        longitude=longitude,
        mlat=mlat,
        edge_index=np.array([sources, targets], dtype=np.int64),
        edge_attr=np.array(attrs, dtype=np.float32),
    )
