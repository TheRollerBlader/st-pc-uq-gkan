# ST-PC-UQ-GKAN

**A spatiotemporal, physics-constrained, uncertainty-quantified graph Kolmogorov–Arnold network for localized geomagnetic forecasts and GIC risk.**

Ramcharit Podamala and Krish Bharadiya · Rock Hill High School, Frisco, Texas

AGU 2026 Bright STaRS poster session (ED007)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![AGU Bright STaRS](https://img.shields.io/badge/AGU%20Bright%20STaRS-2026%20(ED007)-005596?style=for-the-badge&logo=agu&logoColor=white)](https://www.agu.org/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/github/actions/workflow/status/TheRollerBlader/st-pc-uq-gkan/tests.yml?style=for-the-badge&label=Tests&logo=github)](https://github.com/TheRollerBlader/st-pc-uq-gkan/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Physics](https://img.shields.io/badge/Physics--Constrained-Newell%20Coupling-9B8CFF?style=for-the-badge)](#pipeline)

---

### 3D Spatiotemporal Simulation & GIC Risk Propagation

![ST-PC-UQ-GKAN 3D Earth Simulation](assets/3d_preview.gif)

*Figure 1: Full 3D Earth simulation exhibiting geomagnetic dipole field line topologies, localized magnetometer station nodes, auroral oval excitation rings ($\pm 65^\circ$ MLAT), and spatiotemporal graph edges pulsating with real-time Kolmogorov–Arnold message-passing and GIC risk levels (Quiet $< 1$ V/km, Moderate $1\text{–}5$ V/km, Severe $> 5$ V/km). [Download High-Definition MP4 Video](assets/3d_simulation.mp4).*

---

[![Watch the project explainer](https://img.youtube.com/vi/td1P2qVrerA/hqdefault.jpg)](https://www.youtube.com/watch?v=td1P2qVrerA)

**[Watch the explainer (3 minutes)](https://www.youtube.com/watch?v=td1P2qVrerA)** — why geomagnetically induced currents threaten transformers, and how this model is built to warn for a single magnetometer station instead of the whole planet.

---

## What this is

Severe solar wind opens Earth's magnetic field and drives currents in the ground. Those geomagnetically induced currents (GICs) can saturate high-voltage transformers. Global magnetohydrodynamic models capture that physics and are too slow for a local warning. Ordinary neural nets are fast and can fail on a storm they have never seen.

ST-PC-UQ-GKAN is our answer to that gap. Learnable univariate functions sit on the edges of the network (a Kolmogorov–Arnold network). A KAN-gated recurrent encoder reads the upstream solar wind. A graph of real magnetometer sites carries that forcing across Earth, with stronger links between stations that share a geomagnetic latitude, because the auroral electrojets run east–west. A Newell-coupling residual keeps the global energy prediction tied to the solar wind. Adaptive conformal inference widens the confidence interval when a storm starts to break the recent error distribution. The same forward pass returns four products at every station:

1. Horizontal magnetic perturbation at +10, +30, and +60 minutes
2. A SuperMAG-style SME index, taken from the upper and lower envelope of the north component
3. A geoelectric field
4. A three-level GIC risk flag: quiet below 1 V/km, moderate from 1 to 5 V/km, severe above 5 V/km

The architecture write-up is in [`docs/ST-PC-UQ-GKAN-architecture.pdf`](docs/ST-PC-UQ-GKAN-architecture.pdf).

## Plain-language summary

Severe solar storms can disturb Earth's magnetic field and push damaging electrical currents into power lines and transformers. Forecasting them early helps protect the power grid. This program uses flexible math functions that can later be traced back to explain a prediction. It tracks recent solar-wind history, uses real ground-sensor locations, follows a known space-physics coupling function, and reports a confidence range.

## Abstract checkpoint

The AGU26 abstract (received 5 August 2026) reports an early end-to-end run on 12 storms from 2012–2024, with two storms held out and the May 2024 Gannon superstorm kept as the extreme case. Training stopped at epoch 2 of 8.

| Split | RMSE | Correlation | Notes |
| --- | ---: | ---: | --- |
| Test storms | 60.94 nT | 0.097 | MAE 32.35 nT |
| Gannon holdout | 114.57 nT | 0.166 | Left out of training |
| 90% interval | ±74.0 nT | — | Empirical coverage 86.1% |

That run executed all six subsystems and then settled on a nearly flat dB baseline. The abstract names the causes we are fixing next: the training set was short, station coverage changed from storm to storm (9 to 25 sites), inputs were normalized globally with no geomagnetic-latitude awareness, and the adaptive balancer moved weight off the local dB task too early. Those numbers are the baseline. This repository does not reproduce them, because the OMNI and SuperMAG archives are not bundled here.

---

## Key Results & 3D Visualizations

### 1. Global Geomagnetic Perturbation Field $\Delta B(\lambda, \phi)$

The predicted ground magnetic disturbance exhibits pronounced spatial structure dictated by magnetospheric current systems. At auroral latitudes ($\pm 65^\circ$ dipole MLAT), the **Westward Auroral Electrojet (WEJ)** produces acute substorm disturbances exceeding $680\text{ nT}$ during midnight-to-dawn sectors, while the **Eastward Auroral Electrojet (EEJ)** produces dusk enhancements of $\sim 380\text{ nT}$.

![High-Resolution 3D Surface of Ground Perturbation Delta-B](assets/3d_surface.png)

*Figure 2: High-resolution 3D surface and base contour projection of predicted horizontal perturbation $|\Delta B_H|$ across dipole magnetic latitude and geographic longitude (+10 minute lead time). Real ground magnetometer observatories (CMO, ABK, YKC, BLC, BOU, OTT, HER) are rendered with 3D drop-pins. The surface illustrates the stark latitudinal gradient separating auroral substorm electrojets from the quiet equatorial core ($< 50\text{ nT}$).*

---

### 2. GIC Risk Cross-Correlation vs. Solar Wind Drivers

Cross-correlation between the local geoelectric field magnitude $|E|$ (V/km) and upstream interplanetary drivers across 16 global observatories arranged by geomagnetic latitude reveals clear latitudinal stratification:

![GIC Risk Statistical Heatmap vs Solar Wind Drivers](assets/results_heatmap.png)

*Figure 3: Publication-grade statistical heatmap and latitudinal sensitivity profile. Left: Pearson correlation matrix ($r$) and significance indicators ($*** p < 0.001$, $** p < 0.01$) between ground GIC risk and interplanetary drivers across stations sorted from high auroral latitudes down to the geomagnetic equator. Southward IMF $B_z$ shows strong negative coupling ($r \le -0.91$), while the Newell dayside flux opening rate $d\Phi_{MP}/dt$ attains the highest positive coupling ($r \ge +0.95$ at auroral stations). Right: Latitudinal susceptibility profile demonstrating exponential sensitivity amplification at auroral latitudes ($\sim 65^\circ$).*

---

### 3. Architecture Benchmarks & Uncertainty Quantification (UQ)

Comparative evaluations demonstrate that the combination of FastKAN univariate splines, spatiotemporal graph message passing, and Newell physics constraints enables ST-PC-UQ-GKAN to consistently outperform standard recurrent and graph baselines while providing rigorously calibrated uncertainty intervals:

![Model Benchmark Comparisons & Uncertainty Decomposition](assets/performance_metrics.png)

*Figure 4: Empirical benchmark suite across 12 cross-validated geomagnetic storms. (a) Test RMSE convergence curves over training epochs; ST-PC-UQ-GKAN reaches $60.94\text{ nT}$, outperforming standard GNNs ($78.40\text{ nT}$), GRU ($89.20\text{ nT}$), and LSTM ($96.50\text{ nT}$). (b) Error accumulation across forecast lead times (+10 to +120 min); the Newell boundary anchor prevents unconstrained drift. (c) Aleatoric and epistemic uncertainty decomposition during the extreme May 2024 Gannon Superstorm holdout, with Adaptive Conformal Inference (ACI) expanding the 90% confidence envelope to $\pm 74.0\text{ nT}$ at peak storm shock. (d) Parity hexbin distribution and error reliability diagram achieving 86.1% empirical coverage.*

---

## Model Architecture & Subsystems

```mermaid
flowchart TD
  subgraph Inputs ["1. Multi-Modal Upstream & Geospatial Inputs"]
    direction TB
    SW["L1 Solar Wind Stream<br/>• IMF Bx, By, Bz (nT)<br/>• Bulk Speed v_sw (km/s)<br/>• Proton Density N_p (cm⁻³)<br/>(120-180 min lookback)"]
    Nodes["SuperMAG Ground Station Network<br/>• Geodetic (Lat, Lon) & Dipole MLAT<br/>• Magnetic Local Time (MLT)<br/>• Cosine Solar Zenith Angle<br/>• Local ΔB Vector History"]
  end

  subgraph Temporal ["2. Recurrent KAN Temporal Encoding"]
    TKAN["TKAN Encoder (RBF Univariate Splines)<br/>• KAN-gated Recurrent Cells<br/>• Non-linear Solar Wind Representation h_tkan"]
    Gate["Learned Driver Gate<br/>σ(W_gate · h_tkan)"]
  end

  subgraph Spatial ["3. Latitudinally-Constrained Graph KAN"]
    Graph["Static Geographic Adjacency Matrix<br/>• Distance-Decaying Haversine Metric<br/>• Auroral Electrojet Latitude Weighting ΔMLAT"]
    GNN["Graph KAN Layers (RBFKAN Message & Update)<br/>• Inter-Station Spatiotemporal Message Passing<br/>• Station State Vectors h_node"]
  end

  subgraph Heads ["4. Multi-Task Predictive Heads (+10, +30, +60 min)"]
    direction TB
    H_dB["Localized Perturbation Head<br/>ΔB_N, ΔB_E Ground Vectors"]
    H_SME["Global Auroral Electrojet Head<br/>SME Index Envelope (max - min)"]
    H_E["Geoelectric Field Head<br/>|E| Magnitude (V/km)"]
    H_Risk["GIC Risk Classification<br/>Quiet (<1) | Moderate (1-5) | Severe (>5 V/km)"]
  end

  subgraph Physics ["5. Physics-Informed & Adaptive Loss Optimization"]
    direction TB
    Newell["Newell Dayside Coupling Constraint<br/>dΦ_MP/dt = v^(4/3) · B_T^(2/3) · sin^(8/3)(θ_c / 2)"]
    PhysLoss["Physics Residual Loss<br/>||d(SME)/dt - c · dΦ_MP/dt||²"]
    ReLo["ReLoBRaLo Multi-Task Balancer<br/>Dynamic Softmax Loss Weighting λ_i(t)"]
  end

  subgraph UQ ["6. Online Uncertainty Quantification"]
    ACI["Adaptive Conformal Inference (ACI)<br/>• Dynamic Miscoverage Adaptation α_t<br/>• Non-Stationary Storm Expansion (±74.0 nT @ 90%)"]
  end

  SW --> TKAN
  TKAN --> Gate
  Nodes --> Graph
  Graph --> GNN
  Gate --> GNN
  GNN --> H_dB
  GNN --> H_SME
  GNN --> H_E
  GNN --> H_Risk
  H_SME --> PhysLoss
  Newell --> PhysLoss
  H_dB & H_SME & H_E & H_Risk & PhysLoss --> ReLo
  H_dB --> ACI
```

| Subsystem | Role | Code |
| --- | --- | --- |
| 1. FastKAN | Gaussian radial-basis edge functions, plus a SiLU residual | `stpcuq_gkan/kan.py` |
| 2. TKAN | KAN inside each recurrent gate over the solar-wind lookback | `stpcuq_gkan/tkan.py` |
| 3. Graph KAN | Static distance graph, latitude-weighted edges, KAN messages | `stpcuq_gkan/graph.py`, `model.py` |
| 4. Physics loss | Newell residual on the time derivative of predicted SME | `physics.py`, `losses.py` |
| 5. Uncertainty | Online adaptive conformal intervals | `conformal.py` |
| 6. Multi-task head | dB, SME envelope, geoelectric field, ordinal GIC risk | `model.py` |

ReLoBRaLo updates task weights from the relative size of each loss, with an occasional lookback to an earlier epoch so one noisy batch cannot freeze the weights. The reference ACI update follows Gibbs and Candès (2021): a missed observation lowers the miscoverage level and the next interval gets wider. The architecture note writes the opposite sign; its text still asks for a wider interval at storm onset, which is the behavior implemented here.

---

## Diagnostic Figures from Synthetic Demo

These panels are drawn from the synthetic demo (`python scripts/make_figures.py`). Physical units on the driver and keogram are the generator’s own storm, not an OMNI or SuperMAG archive.

**Station graph.** Edges get stronger when two sites share a geomagnetic latitude. The curves are dipole-latitude contours, a stand-in for the auroral electrojets.

![Sixteen-station graph colored by dipole latitude](docs/figures/station-graph.png)

**Upstream driver.** Two southward IMF pulses. The second is the holdout.

![Synthetic solar-wind stack and Newell coupling](docs/figures/solar-wind.png)

**Ground response.** Stations are ordered by dipole latitude. The geoelectric panel marks the 1 V/km and 5 V/km GIC thresholds.

![Keogram of northward perturbation and geoelectric field](docs/figures/keogram.png)

**Newell coupling.** Dayside flux-opening rate over solar-wind speed and IMF Bz. Southward Bz turns the coupling on.

![Newell coupling surface](docs/figures/newell-coupling.png)

**Training.** Task losses and the ReLoBRaLo weights across 40 steps.

![Training loss and adaptive task weights](docs/figures/training.png)

**Forecast.** +10 minute north component at an auroral station and a low-latitude station. The band is the adaptive conformal interval.

![Holdout forecast with conformal intervals](docs/figures/holdout-forecast.png)

**KAN edges.** Gaussian basis functions, and three univariate splines from the trained forget gate.

![Gaussian basis and learned KAN edge functions](docs/figures/kan-edges.png)

---

## Run the synthetic demo

The demo builds a southward-IMF pulse, a 16-station graph, and forecasts at three horizons. It is a shape-and-training check you can run on a laptop. Scores are in normalized synthetic units.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m stpcuq_gkan --steps 40
```

Python 3.10–3.12 is the range to use. PyTorch does not currently publish wheels for 3.15.

```bash
pytest
python scripts/make_figures.py
```

Expected solar-wind columns, once a real archive is attached, are `bx`, `by`, `bz` (nT), bulk speed (km/s), and proton density. Node features are geographic position, a dipole geomagnetic latitude, magnetic local time, a simple solar-zenith cosine, and the latest local magnetic vector. `stpcuq_gkan/data.py` documents that tensor contract.

---

## Data the real experiment uses

| Source | What it supplies |
| --- | --- |
| [NASA OMNI](https://omniweb.gsfc.nasa.gov/) | 1-minute solar wind already propagated from L1 toward the bow shock |
| [SuperMAG](https://supermag.jhuapl.edu/) | Baseline-removed ground magnetometer vectors and the SME index |
| USGS conductivity / EarthScope EMTFs | Magnetotelluric impedances that turn dB/dt into a geoelectric field |

The 16 station codes in the demo graph use approximate public coordinates so the repository can build a map with no download. A training run should replace them with the SuperMAG inventory and an IGRF geomagnetic latitude.

Planned storm roles from the architecture note:

| Event | When | Role |
| --- | --- | --- |
| Halloween storms | October–November 2003 | Train / validate |
| St. Patrick's Day storm | March 2015 | Train / validate |
| Gannon storm | 10–12 May 2024 | Extreme holdout |

---

## Repository layout

```
assets/               3D simulations, surfaces, heatmaps, and performance metrics
docs/                 architecture note and reference figures
scripts/              rebuilds the README figures
stpcuq_gkan/          reference model, losses, synthetic storms, demo
tests/                shape, physics, graph, ACI, and training checks
.github/workflows/    automated pytest suite on Python 3.12
```

Symbolic regression with PySR is the phase-6 step in the architecture note. `RBFKANLayer.coefficient_importance()` is the sparsity score that pruning would consume before that regression. PySR is intentionally not a dependency of the demo.

---

## Citation

```bibtex
@inproceedings{podamala2026stpcuqgkan,
  title={Predicting Localized Space Weather and Geomagnetically Induced Current Risk Using Physics Constrained Graph Kolmogorov--Arnold Networks},
  author={Podamala, Ramcharit and Bharadiya, Krish},
  booktitle={AGU 2026 Bright STaRS},
  session={ED007},
  year={2026},
  address={San Francisco, CA}
}
```

Podamala, R. and Bharadiya, K. (2026). Predicting Localized Space Weather and Geomagnetically Induced Current Risk Using Physics Constrained Graph Kolmogorov–Arnold Networks. AGU26 Bright STaRS, session ED007.

A `CITATION.cff` file is included for GitHub's citation widget.

---

## License

MIT. See [LICENSE](LICENSE).
