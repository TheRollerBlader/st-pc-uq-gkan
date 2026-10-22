"""Render the README figures from the synthetic storm and a short training run.

    python scripts/make_figures.py

Outputs land in docs/figures. They are demo diagnostics, not the AGU storm scores.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from stpcuq_gkan.conformal import AdaptiveConformalInference
from stpcuq_gkan.data import synthetic_clock, synthesize_storms
from stpcuq_gkan.graph import dipole_magnetic_latitude
from stpcuq_gkan.model import ModelConfig, STPCUQGKAN
from stpcuq_gkan.physics import newell_coupling_numpy
from stpcuq_gkan.train import fit

OUT = ROOT / "docs" / "figures"

BG = "#070B14"
PANEL = "#0E1626"
INK = "#E7F0FA"
MUTED = "#8EA3BA"
CYAN = "#3EE0FF"
AMBER = "#FFB020"
MAGENTA = "#FF4D8D"
MINT = "#3DDC97"
VIOLET = "#9B8CFF"
GRID = "#1B2B44"
SPINE = "#2A3E5C"

STORM = LinearSegmentedColormap.from_list(
    "storm",
    ["#07111F", "#0E3A5C", "#1FA2C4", "#F2C14E", "#FFF4D6"],
)
COUPLING = LinearSegmentedColormap.from_list(
    "coupling",
    ["#10192B", "#16324F", "#1C7C9C", "#E39B2B", "#FFE7A3"],
)
LATITUDE = LinearSegmentedColormap.from_list(
    "mlat",
    ["#FF4D8D", "#F2C14E", "#8EA3BA", "#3EE0FF", "#9B8CFF"],
)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": PANEL,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": SPINE,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.titlesize": 14,
            "axes.titleweight": "regular",
            "axes.labelsize": 10,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.titlelocation": "left",
        }
    )


def _mono(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("DejaVu Sans Mono")
        label.set_fontsize(8)


def _footer(fig: plt.Figure, note: str) -> None:
    fig.text(
        0.012,
        0.012,
        note,
        color=MUTED,
        fontsize=8,
        fontfamily="DejaVu Sans Mono",
        ha="left",
        va="bottom",
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print(f"{path.relative_to(ROOT)}  {path.stat().st_size // 1024} KB")


def _label(ax: plt.Axes, x: float, y: float, text: str, color: str = INK) -> None:
    artist = ax.text(x, y, text, color=color, fontsize=8, fontfamily="DejaVu Sans Mono", zorder=5)
    artist.set_path_effects([pe.withStroke(linewidth=2.8, foreground=BG)])


def station_graph() -> None:
    clock = synthetic_clock()
    graph = clock.graph
    fig = plt.figure(figsize=(12.4, 6.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 0.62], wspace=0.06)
    ax = fig.add_subplot(grid[0])
    side = fig.add_subplot(grid[1])

    glon = np.linspace(-180, 180, 361)
    glat = np.linspace(-90, 90, 181)
    mesh_lon, mesh_lat = np.meshgrid(glon, glat)
    mlat_grid = dipole_magnetic_latitude(mesh_lat, mesh_lon)
    shade = 0.55 + 0.45 * np.cos(np.radians(mesh_lat))
    ax.pcolormesh(
        mesh_lon,
        mesh_lat,
        shade,
        cmap=LinearSegmentedColormap.from_list("ocean", ["#070B14", "#12263A"]),
        shading="auto",
        zorder=0,
    )
    ax.contour(
        mesh_lon,
        mesh_lat,
        np.abs(mlat_grid),
        levels=[55, 70],
        colors=[CYAN, VIOLET],
        linewidths=[0.7, 0.9],
        alpha=0.85,
        zorder=1,
    )
    ax.axhline(0, color=GRID, lw=0.6, zorder=1)

    lon = graph.longitude
    lat = graph.latitude
    sources, targets = graph.edge_index
    seen: set[tuple[int, int]] = set()
    segments: list[np.ndarray] = []
    colors: list[float] = []
    for src, dst, attr in zip(sources, targets, graph.edge_attr):
        pair = (int(src), int(dst)) if src < dst else (int(dst), int(src))
        if pair in seen:
            continue
        seen.add(pair)
        strength = float(attr[2])
        pieces = _wrap(lon[src], lat[src], lon[dst], lat[dst])
        for piece in pieces:
            segments.append(piece)
            colors.append(strength)
    rgba = []
    widths = []
    base = np.array(matplotlib.colors.to_rgb(CYAN))
    for strength in colors:
        rgba.append((*base, 0.08 + 0.85 * strength))
        widths.append(0.35 + 2.4 * strength)
    collection = LineCollection(segments, colors=rgba, linewidths=widths, zorder=2)
    ax.add_collection(collection)

    norm = plt.Normalize(vmin=-70, vmax=70)
    ax.scatter(
        lon,
        lat,
        s=46,
        c=graph.mlat,
        cmap=LATITUDE,
        norm=norm,
        edgecolors="#F4F8FC",
        linewidths=0.6,
        zorder=4,
    )
    nudges = {
        "CMO": (-28, 8),
        "BLC": (6, 8),
        "YKC": (-34, 10),
        "FCC": (10, -8),
        "MEA": (8, -11),
        "VIC": (-24, -12),
        "NEW": (8, -12),
        "BOU": (8, 8),
        "OTT": (8, 7),
        "FRD": (8, -11),
        "SJG": (8, -11),
        "HON": (-30, 6),
        "GUA": (8, 6),
        "ABK": (8, 6),
        "LER": (8, -11),
        "HER": (8, -12),
    }
    for code, x, y in zip(graph.codes, lon, lat):
        artist = ax.annotate(
            code,
            (x, y),
            textcoords="offset points",
            xytext=nudges.get(code, (6, 6)),
            color=INK,
            fontsize=7.5,
            fontfamily="DejaVu Sans Mono",
            zorder=5,
        )
        artist.set_path_effects([pe.withStroke(linewidth=2.6, foreground=BG)])

    ax.set_xlim(-180, 180)
    ax.set_ylim(-75, 85)
    ax.set_xlabel("Geographic longitude")
    ax.set_ylabel("Geographic latitude")
    ax.set_title("Magnetometer graph  ·  auroral zones at 55° and 70° dipole latitude")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 81, 30))
    ax.grid(True, zorder=0)
    _mono(ax)
    handles = [
        Line2D([0], [0], color=CYAN, lw=1.2, label="|MLAT| = 55°"),
        Line2D([0], [0], color=VIOLET, lw=1.2, label="|MLAT| = 70°"),
    ]
    legend = ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=8)
    for text in legend.get_texts():
        text.set_color(MUTED)

    order = np.argsort(graph.mlat)
    side.barh(
        np.arange(len(order)),
        graph.mlat[order],
        color=LATITUDE(norm(graph.mlat[order])),
        height=0.72,
        zorder=2,
    )
    side.axvline(0, color=GRID, lw=0.7)
    side.axvline(65, color=CYAN, lw=0.7, ls="--")
    side.axvline(-65, color=CYAN, lw=0.7, ls="--")
    side.set_yticks(np.arange(len(order)))
    side.set_yticklabels([graph.codes[i] for i in order])
    side.set_xlabel("Dipole MLAT")
    side.set_title("Latitude order")
    side.set_xlim(-80, 85)
    side.grid(True, axis="x")
    _mono(side)
    _footer(fig, "ST-PC-UQ-GKAN   ·   16 approximate public sites   ·   dipole MLAT, not IGRF")
    _save(fig, "station-graph.png")


def _wrap(lon1: float, lat1: float, lon2: float, lat2: float) -> list[np.ndarray]:
    if abs(lon2 - lon1) <= 180:
        return [np.array([[lon1, lat1], [lon2, lat2]])]
    if lon1 > lon2:
        lon1, lon2 = lon2, lon1
        lat1, lat2 = lat2, lat1
    span = (lon2 + 360) - lon1
    frac = (180 - lon1) / span
    lat_break = lat1 + frac * (lat2 - lat1)
    return [
        np.array([[lon1, lat1], [180, lat_break]]),
        np.array([[-180, lat_break], [lon2, lat2]]),
    ]


def newell_surface() -> None:
    bz = np.linspace(-32, 18, 280)
    speed = np.linspace(300, 900, 220)
    grid_bz, grid_v = np.meshgrid(bz, speed)
    coupling = newell_coupling_numpy(grid_v, np.zeros_like(grid_bz), grid_bz)
    south = coupling[grid_bz < -1]
    vmin = max(float(np.percentile(south, 8)), 1.0)
    vmax = float(np.percentile(south, 99.5))

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    image = ax.pcolormesh(
        bz,
        speed,
        coupling,
        cmap=COUPLING,
        shading="auto",
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.contour(
        bz,
        speed,
        coupling,
        levels=np.geomspace(vmin, vmax, 5)[1:-1],
        colors="#F4F8FC",
        linewidths=0.55,
        alpha=0.45,
    )
    ax.axvline(0, color=INK, lw=0.8, ls="--", alpha=0.7)
    quiet = (3.0, 370.0)
    storm = (-25.0, 690.0)
    ax.scatter([quiet[0], storm[0]], [quiet[1], storm[1]], s=36, c=[MINT, MAGENTA], zorder=4, edgecolors=INK, linewidths=0.4)
    _label(ax, quiet[0] + 1.2, quiet[1] + 18, "quiet wind", MINT)
    _label(ax, storm[0] + 1.2, storm[1] + 18, "storm pulse", MAGENTA)
    ax.text(8.2, 860, "northward IMF", color=MUTED, fontsize=8, ha="left")
    ax.text(-30.5, 860, "southward IMF", color=AMBER, fontsize=8, ha="left")
    bar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.046)
    bar.set_label("Newell coupling", color=MUTED)
    bar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(bar.ax.get_yticklabels(), color=MUTED, fontname="DejaVu Sans Mono", fontsize=8)
    bar.outline.set_edgecolor(SPINE)
    ax.set_xlabel(r"IMF $B_z$  (nT)")
    ax.set_ylabel(r"Solar-wind speed  (km/s)")
    ax.set_title(r"Newell dayside coupling    $v^{4/3}\, B_T^{2/3}\, \sin^{8/3}(\theta_c/2)$    ·    $B_y = 0$")
    ax.set_xlim(bz.min(), bz.max())
    ax.set_ylim(speed.min(), speed.max())
    _mono(ax)
    _footer(fig, "ST-PC-UQ-GKAN   ·   physics residual used in the training loss   ·   By fixed at 0 nT")
    _save(fig, "newell-coupling.png")


def solar_wind(clock) -> None:
    minute = clock.minute
    mid = clock.origin0 + clock.hold_start
    left = int(np.argmax(np.where(minute < mid, clock.storm, -1)))
    right = int(np.argmax(np.where(minute >= mid, clock.storm, -1)))
    panels = [
        (clock.bz, "IMF $B_z$", "nT", AMBER),
        (clock.by, "IMF $B_y$", "nT", VIOLET),
        (clock.speed, "Bulk speed", "km/s", CYAN),
        (clock.density, "Proton density", "cm$^{-3}$", MINT),
        (clock.coupling / clock.coupling.max(), "Newell coupling", "relative", MAGENTA),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12.2, 8.2), sharex=True)
    for ax, (series, name, unit, color) in zip(axes, panels):
        ax.fill_between(minute, series, series.min(), color=color, alpha=0.13, linewidth=0)
        ax.plot(minute, series, color=color, lw=1.35)
        ax.axvline(minute[left], color=CYAN, lw=0.7, ls="--", alpha=0.8)
        ax.axvline(minute[right], color=MAGENTA, lw=0.7, ls="--", alpha=0.8)
        ax.set_ylabel(unit)
        ax.set_title(name, loc="left", fontsize=11, color=color, pad=2)
        ax.grid(True, axis="y")
        ax.margins(x=0)
        _mono(ax)
    axes[0].text(minute[left] + 2, axes[0].get_ylim()[1], "train", color=CYAN, fontsize=8, va="top")
    axes[0].text(minute[right] + 2, axes[0].get_ylim()[1], "holdout", color=MAGENTA, fontsize=8, va="top")
    axes[-1].set_xlabel("Minute")
    _footer(fig, "ST-PC-UQ-GKAN   ·   synthetic L1 driver   ·   two southward pulses, one withheld")
    _save(fig, "solar-wind.png")


def keogram(clock) -> None:
    order = np.argsort(clock.graph.mlat)
    codes = [clock.graph.codes[i] for i in order]
    minute = clock.minute
    bn = clock.bn[order]
    electric = clock.e_field[order]
    fig, axes = plt.subplots(2, 1, figsize=(12.2, 7.4), sharex=True, height_ratios=[1.15, 1])
    bn_image = axes[0].imshow(
        bn,
        aspect="auto",
        origin="lower",
        cmap=STORM,
        extent=[minute[0], minute[-1], -0.5, len(codes) - 0.5],
        interpolation="nearest",
    )
    e_image = axes[1].imshow(
        electric,
        aspect="auto",
        origin="lower",
        cmap=STORM,
        extent=[minute[0], minute[-1], -0.5, len(codes) - 0.5],
        interpolation="nearest",
    )
    y = np.arange(len(codes))
    axes[1].contour(
        minute,
        y,
        electric,
        levels=[1, 5],
        colors=[AMBER, MAGENTA],
        linewidths=[0.8, 1.05],
    )
    for ax, title in zip(axes, ["Northward ground perturbation", "Geoelectric field and GIC thresholds"]):
        ax.set_yticks(y)
        ax.set_yticklabels(codes)
        ax.set_title(title)
        _mono(ax)
    axes[1].set_xlabel("Minute")
    bar0 = fig.colorbar(bn_image, ax=axes[0], pad=0.012, fraction=0.035)
    bar1 = fig.colorbar(e_image, ax=axes[1], pad=0.012, fraction=0.035)
    bar0.set_label("nT", color=MUTED)
    bar1.set_label("V/km", color=MUTED)
    for bar in (bar0, bar1):
        bar.outline.set_edgecolor(SPINE)
        plt.setp(bar.ax.get_yticklabels(), color=MUTED, fontsize=8, fontname="DejaVu Sans Mono")
    threshold_key = [
        Line2D([0], [0], color=AMBER, lw=1.4, label="1 V/km moderate"),
        Line2D([0], [0], color=MAGENTA, lw=1.4, label="5 V/km severe"),
    ]
    key = axes[1].legend(handles=threshold_key, frameon=False, loc="center", fontsize=8)
    for text in key.get_texts():
        text.set_color(INK)
    _footer(fig, "ST-PC-UQ-GKAN   ·   stations sorted by dipole latitude   ·   synthetic ground response")
    _save(fig, "keogram.png")


def training_and_forecast(clock) -> None:
    torch.manual_seed(0)
    train, holdout = synthesize_storms(160, 16, seed=0)
    model = STPCUQGKAN(ModelConfig(hidden=32, basis=6))
    history = fit(model, train, steps=40, seed=0)
    model.eval()
    with torch.no_grad():
        trained = model(train.solar_wind, train.nodes, train.edge_index, train.edge_attr)
        prediction = model(holdout.solar_wind, holdout.nodes, holdout.edge_index, holdout.edge_attr)
    pred = prediction["db"][:, :, 0, 0].cpu().numpy()
    truth = holdout.db[:, :, 0, 0].cpu().numpy()
    pred_train = trained["db"][:, :, 0, 0].cpu().numpy()
    truth_train = train.db[:, :, 0, 0].cpu().numpy()

    steps = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    axes[0].plot(steps, [row["total"] for row in history], color=INK, lw=2.0, label="total")
    for name, color in (("db", CYAN), ("sme", AMBER), ("efield", MINT), ("risk", MAGENTA), ("physics", VIOLET)):
        axes[0].plot(steps, [row[name] for row in history], color=color, lw=1.15, label=name)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Five-task loss over 40 steps")
    axes[0].grid(True)
    legend = axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    for text in legend.get_texts():
        text.set_color(MUTED)
    _mono(axes[0])

    names = ["db", "sme", "efield", "risk", "physics"]
    colors = [CYAN, AMBER, MINT, MAGENTA, VIOLET]
    stack = np.vstack([[row[f"w_{name}"] for row in history] for name in names])
    axes[1].stackplot(steps, stack, colors=colors, alpha=0.92, labels=names)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Task weight")
    axes[1].set_title("ReLoBRaLo weights")
    axes[1].grid(True, axis="y")
    legend = axes[1].legend(
        frameon=False, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18)
    )
    for text in legend.get_texts():
        text.set_color(INK)
    _mono(axes[1])
    _footer(fig, "ST-PC-UQ-GKAN   ·   synthetic training run   ·   weights stay a distribution")
    _save(fig, "training.png")

    graph = holdout.graph
    auroral = int(np.argmin(np.abs(np.abs(graph.mlat) - 65)))
    equator = int(np.argmin(np.abs(graph.mlat)))
    fig, axes = plt.subplots(3, 1, figsize=(12.2, 8.4), sharey=False, gridspec_kw={"hspace": 0.46})
    panels = [
        (axes[0], truth_train[:, auroral], pred_train[:, auroral], None, f"Train  ·  {graph.codes[auroral]}  ·  dipole MLAT {graph.mlat[auroral]:.0f}°", CYAN),
        (axes[1], truth[:, auroral], pred[:, auroral], True, f"Holdout  ·  {graph.codes[auroral]}  ·  dipole MLAT {graph.mlat[auroral]:.0f}°", MAGENTA),
        (axes[2], truth[:, equator], pred[:, equator], True, f"Holdout  ·  {graph.codes[equator]}  ·  dipole MLAT {graph.mlat[equator]:.0f}°", AMBER),
    ]
    for ax, observed, forecast, use_aci, title, color in panels:
        time = np.arange(len(observed))
        if use_aci:
            aci = AdaptiveConformalInference(gamma=0.08)
            widths = []
            fallback = float(np.std(observed) + 1e-3)
            for value in np.abs(forecast - observed):
                widths.append(aci.half_width(fallback))
                aci.update(float(value), fallback)
            band = np.array(widths)
            ax.fill_between(time, forecast - band, forecast + band, color=CYAN, alpha=0.16, linewidth=0, label="ACI interval")
        ax.plot(time, observed, color=INK, lw=1.55, label="target")
        ax.plot(time, forecast, color=color, lw=1.5, label="forecast +10 min")
        ax.set_title(title)
        ax.set_ylabel("normalized ΔB")
        ax.grid(True, axis="y")
        legend = ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
        for text in legend.get_texts():
            text.set_color(MUTED)
        _mono(ax)
    axes[-1].set_xlabel("Window")
    _footer(fig, "ST-PC-UQ-GKAN   ·   +10 minute north component   ·   synthetic normalized units")
    _save(fig, "holdout-forecast.png")

    layer = model.encoder.cell.forget
    importance = layer.coefficient_importance().mean(dim=0).cpu().numpy()
    chosen = np.argsort(importance)[-3:][::-1]
    xs = torch.linspace(-2.2, 2.2, 400)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    probe = xs.view(-1, 1)
    bases = layer.basis(probe.expand(-1, layer.in_features).contiguous())[:, 0].detach().cpu().numpy()
    for index in range(bases.shape[1]):
        axes[0].plot(xs.numpy(), bases[:, index], color=plt.cm.GnBu(0.35 + 0.1 * index), lw=1.3)
    axes[0].set_title("Gaussian radial basis on one edge")
    axes[0].set_xlabel("tanh-scaled input")
    axes[0].set_ylabel("basis response")
    axes[0].grid(True)
    _mono(axes[0])

    palette = [CYAN, AMBER, MAGENTA]
    grid = xs.numpy()
    with torch.no_grad():
        for _slot, (in_index, color) in enumerate(zip(chosen, palette)):
            out_index = int(torch.argmax(layer.coefficient_importance()[:, int(in_index)]))
            coeff = layer.coeff[out_index, int(in_index)].detach().cpu().numpy()
            sample = torch.zeros(xs.numel(), layer.in_features)
            sample[:, int(in_index)] = xs
            curve = layer.basis(sample)[:, int(in_index)].cpu().numpy() @ coeff
            axes[1].plot(grid, curve, color=color, lw=1.8, label=f"edge {int(in_index)} → {out_index}")
    axes[1].axhline(0, color=GRID, lw=0.7)
    low, high = axes[1].get_ylim()
    axes[1].set_ylim(low, high + 0.28 * (high - low))
    axes[1].set_title("Learned spline on the strongest edges")
    axes[1].set_xlabel("input value")
    axes[1].set_ylabel("spline response")
    axes[1].grid(True)
    legend = axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    for text in legend.get_texts():
        text.set_color(MUTED)
    _mono(axes[1])
    _footer(fig, "ST-PC-UQ-GKAN   ·   FastKAN forget gate after the synthetic run   ·   univariate edge functions")
    _save(fig, "kan-edges.png")


def main() -> None:
    _style()
    clock = synthetic_clock(160, 16, seed=0)
    station_graph()
    newell_surface()
    solar_wind(clock)
    keogram(clock)
    training_and_forecast(clock)


if __name__ == "__main__":
    main()
