import numpy as np
import torch

from stpcuq_gkan.conformal import AdaptiveConformalInference
from stpcuq_gkan.data import synthesize_storms
from stpcuq_gkan.graph import build_station_graph
from stpcuq_gkan.kan import RBFKANLayer
from stpcuq_gkan.losses import ReLoBRaLo, task_losses
from stpcuq_gkan.model import ModelConfig, STPCUQGKAN
from stpcuq_gkan.physics import gic_risk_level, newell_coupling
from stpcuq_gkan.tkan import TKANEncoder
from stpcuq_gkan.train import training_step


def test_newell_coupling_prefers_southward_imf():
    speed = torch.tensor([500.0, 500.0])
    by = torch.zeros(2)
    bz = torch.tensor([10.0, -10.0])
    coupling = newell_coupling(speed, by, bz)
    assert coupling[1] > coupling[0] * 5


def test_risk_thresholds():
    field = torch.tensor([0.2, 1.0, 5.0, 5.1])
    assert gic_risk_level(field).tolist() == [0, 1, 1, 2]


def test_graph_prefers_shared_magnetic_latitude():
    graph = build_station_graph(k=3)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == 3
    sources, targets = graph.edge_index
    edge_gap = np.abs(graph.mlat[sources] - graph.mlat[targets]).mean()
    all_gap = np.abs(graph.mlat[:, None] - graph.mlat[None, :])
    all_gap = all_gap[np.triu_indices(len(graph.mlat), k=1)].mean()
    assert edge_gap < all_gap


def test_kan_and_tkan_shapes_and_gradients():
    layer = RBFKANLayer(4, 3, num_basis=4)
    x = torch.randn(5, 4, requires_grad=True)
    y = layer(x)
    assert y.shape == (5, 3)
    y.sum().backward()
    assert x.grad is not None
    assert layer.coefficient_importance().shape == (3, 4)

    encoder = TKANEncoder(5, 7, num_basis=4)
    encoded = encoder(torch.randn(2, 6, 5))
    assert encoded.shape == (2, 7)


def test_relobralo_weights_stay_a_distribution():
    balancer = ReLoBRaLo(mix=0.5, lookback_probability=0.0)
    losses = {name: torch.tensor(float(index + 1), requires_grad=True) for index, name in enumerate(("db", "sme", "efield", "risk", "physics"))}
    total = balancer.combine(losses)
    total.backward()
    assert torch.allclose(balancer.lambdas.sum(), torch.tensor(1.0), atol=1e-5)
    assert total.ndim == 0


def test_aci_widens_after_misses():
    aci = AdaptiveConformalInference(gamma=0.2)
    before = aci.half_width(fallback=1.0)
    for residual in (5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0):
        aci.update(residual, fallback=1.0)
    assert aci.half_width(fallback=1.0) > before
    assert aci.alpha < 0.1


def test_forward_and_one_training_step():
    torch.manual_seed(0)
    train, _holdout = synthesize_storms(n_windows=12, lookback=8, seed=1)
    model = STPCUQGKAN(ModelConfig(hidden=8, basis=4, graph_layers=1))
    prediction = model(train.solar_wind, train.nodes, train.edge_index, train.edge_attr)
    assert prediction["db"].shape == train.db.shape
    assert prediction["sme"].shape == train.sme.shape
    assert prediction["e_field"].shape == train.e_field.shape
    assert prediction["risk_logits"].shape[:3] == train.risk.shape

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    report = training_step(model, ReLoBRaLo(), train, optimizer)
    assert np.isfinite(report["total"])
    losses = task_losses(
        prediction,
        {
            "db": train.db,
            "sme": train.sme,
            "e_field": train.e_field,
            "risk": train.risk,
            "newell": train.newell,
        },
        model.coupling_scale,
    )
    assert set(losses) == {"db", "sme", "efield", "risk", "physics"}
