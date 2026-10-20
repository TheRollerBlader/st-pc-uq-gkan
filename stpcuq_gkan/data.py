"""Synthetic storm windows with the same tensors a real OMNI/SuperMAG run would use.

The generator builds a southward-IMF pulse, converts it to a Newell coupling
series, and paints a stronger ground response on stations near 65° dipole
latitude. Targets at +10, +30, and +60 minutes are read off that series.
Real training data would replace this file's arrays with OMNI solar wind and
baseline-removed SuperMAG vectors; the tensor contract stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from stpcuq_gkan.graph import StationGraph, build_station_graph
from stpcuq_gkan.physics import gic_risk_level_numpy, newell_coupling_numpy

SOLAR_COLUMNS = ("bx", "by", "bz", "speed", "density")
NODE_COLUMNS = (
    "lat",
    "lon_sin",
    "lon_cos",
    "mlat",
    "mlt_sin",
    "mlt_cos",
    "cos_sza",
    "bn",
    "be",
    "bz",
)


@dataclass
class StormBatch:
    solar_wind: Tensor
    nodes: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    db: Tensor
    sme: Tensor
    e_field: Tensor
    risk: Tensor
    newell: Tensor
    graph: StationGraph

    def to(self, device: torch.device | str) -> "StormBatch":
        return StormBatch(
            solar_wind=self.solar_wind.to(device),
            nodes=self.nodes.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            db=self.db.to(device),
            sme=self.sme.to(device),
            e_field=self.e_field.to(device),
            risk=self.risk.to(device),
            newell=self.newell.to(device),
            graph=self.graph,
        )


def _scale(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=tuple(range(train.ndim - 1)), keepdims=True)
    std = train.std(axis=tuple(range(train.ndim - 1)), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return tuple((array - mean) / std for array in (train, *others))


def synthesize_storms(
    n_windows: int = 48,
    lookback: int = 24,
    horizons: tuple[int, ...] = (10, 30, 60),
    seed: int = 0,
) -> tuple[StormBatch, StormBatch]:
    """Return consecutive train and holdout windows. Holdout is the last quarter."""
    rng = np.random.default_rng(seed)
    graph = build_station_graph()
    horizon_max = max(horizons)
    total = n_windows + lookback + horizon_max
    clock = np.arange(total, dtype=np.float64)
    gap = max(horizons)
    train_count = n_windows // 2
    hold_start = train_count + gap
    if hold_start + 4 >= n_windows:
        train_count = max(int(n_windows * 0.75), 2)
        hold_start = train_count

    def _pulse(center: float, width: float) -> np.ndarray:
        return np.exp(-0.5 * ((clock - center) / width) ** 2)

    # One pulse in the training block and a similar pulse in the holdout.
    # The embargo between them is one full forecast horizon, so a training
    # target at +60 minutes does not land on the holdout storm.
    origin0 = lookback - 1
    train_center = origin0 + max(train_count * 0.45, 1.0)
    hold_span = max(n_windows - hold_start, 1)
    hold_center = origin0 + hold_start + hold_span * 0.45
    storm = 0.9 * _pulse(train_center, 8.0) + _pulse(hold_center, 8.0)
    bx = np.full(total, 1.5)
    by = 5.0 * np.sin(clock / 7.0)
    bz = 3.0 - 28.0 * storm
    speed = 370.0 + 320.0 * storm
    density = 4.0 + 10.0 * storm
    coupling = newell_coupling_numpy(speed, by, bz)
    coupling_n = coupling / (coupling.max() + 1e-6)

    gain = np.exp(-0.5 * ((np.abs(graph.mlat) - 65.0) / 12.0) ** 2)
    gain = 0.15 + 0.85 * gain
    bn = gain[:, None] * (80.0 + 520.0 * coupling_n[None, :])
    be = gain[:, None] * (20.0 * np.sin(clock / 11.0)[None, :])
    bz_ground = 0.1 * bn
    bn = bn + rng.normal(0.0, 4.0, size=bn.shape)
    be = be + rng.normal(0.0, 2.0, size=be.shape)

    e_series = np.clip(gain[:, None] * (0.2 + 7.5 * coupling_n[None, :]), 0.0, None)
    north_envelope = bn.max(axis=0) - bn.min(axis=0)

    origins = np.arange(lookback - 1, lookback - 1 + n_windows)
    solar = []
    nodes = []
    db = []
    sme = []
    e_field = []
    risk = []
    newell = []
    lat = graph.latitude / 90.0
    lon_rad = np.radians(graph.longitude)
    mlat = graph.mlat / 90.0

    for origin in origins:
        history = slice(origin - lookback + 1, origin + 1)
        solar.append(
            np.stack(
                [
                    bx[history],
                    by[history],
                    bz[history],
                    speed[history],
                    density[history],
                ],
                axis=-1,
            )
        )
        ut = float(origin % 24)
        mlt = (ut + graph.longitude / 15.0) % 24.0
        hour = np.radians(15.0 * (ut - 12.0) + graph.longitude)
        cos_sza = np.cos(np.radians(graph.latitude)) * np.cos(hour)
        nodes.append(
            np.stack(
                [
                    lat,
                    np.sin(lon_rad),
                    np.cos(lon_rad),
                    mlat,
                    np.sin(2 * np.pi * mlt / 24.0),
                    np.cos(2 * np.pi * mlt / 24.0),
                    cos_sza,
                    bn[:, origin] / 200.0,
                    be[:, origin] / 50.0,
                    bz_ground[:, origin] / 200.0,
                ],
                axis=-1,
            )
        )
        future = [origin + step for step in horizons]
        db.append(
            np.stack(
                [np.stack([bn[:, step], be[:, step]], axis=-1) for step in future],
                axis=1,
            )
        )
        sme.append(north_envelope[list(future)])
        e_field.append(e_series[:, future])
        risk.append(gic_risk_level_numpy(e_series[:, future]))
        newell.append(coupling[origin])

    solar_arr = np.stack(solar).astype(np.float32)
    node_arr = np.stack(nodes).astype(np.float32)
    db_arr = np.stack(db).astype(np.float32)
    sme_arr = np.stack(sme).astype(np.float32)
    e_arr = np.stack(e_field).astype(np.float32)
    risk_arr = np.stack(risk).astype(np.int64)
    newell_arr = np.array(newell, dtype=np.float32)

    split = train_count
    if hold_start > split:
        scaled = _scale(
            solar_arr[:split],
            solar_arr[split:hold_start],
            solar_arr[hold_start:],
        )
    else:
        scaled = _scale(solar_arr[:split], solar_arr[hold_start:])
    solar_all = np.concatenate(scaled)
    # Node location channels are already O(1). Scale only the magnetic channels
    # with training statistics, then keep the geographic channels as built.
    mag_train = node_arr[:split, :, 7:]
    mag_mean = mag_train.mean(axis=(0, 1), keepdims=True)
    mag_std = np.maximum(mag_train.std(axis=(0, 1), keepdims=True), 1e-6)
    node_arr = node_arr.copy()
    node_arr[:, :, 7:] = (node_arr[:, :, 7:] - mag_mean) / mag_std

    db_mean = db_arr[:split].mean()
    db_std = max(float(db_arr[:split].std()), 1e-6)
    db_arr = (db_arr - db_mean) / db_std
    sme_mean = sme_arr[:split].mean()
    sme_std = max(float(sme_arr[:split].std()), 1e-6)
    sme_arr = (sme_arr - sme_mean) / sme_std

    def pack(sl: slice) -> StormBatch:
        return StormBatch(
            solar_wind=torch.from_numpy(solar_all[sl]),
            nodes=torch.from_numpy(node_arr[sl]),
            edge_index=torch.from_numpy(graph.edge_index),
            edge_attr=torch.from_numpy(graph.edge_attr),
            db=torch.from_numpy(db_arr[sl]),
            sme=torch.from_numpy(sme_arr[sl]),
            e_field=torch.from_numpy(e_arr[sl]),
            risk=torch.from_numpy(risk_arr[sl]),
            newell=torch.from_numpy(newell_arr[sl]),
            graph=graph,
        )

    return pack(slice(0, split)), pack(slice(hold_start, None))
