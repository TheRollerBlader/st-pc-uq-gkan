"""One optimization step and a short synthetic training loop."""

from __future__ import annotations

import torch
from torch import nn

from stpcuq_gkan.data import StormBatch
from stpcuq_gkan.losses import LOSS_NAMES, ReLoBRaLo, task_losses
from stpcuq_gkan.model import STPCUQGKAN


def training_step(
    model: STPCUQGKAN,
    balancer: ReLoBRaLo,
    batch: StormBatch,
    optimizer: torch.optim.Optimizer,
    stride_minutes: float = 1.0,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch.solar_wind, batch.nodes, batch.edge_index, batch.edge_attr)
    losses = task_losses(
        prediction,
        {
            "db": batch.db,
            "sme": batch.sme,
            "e_field": batch.e_field,
            "risk": batch.risk,
            "newell": batch.newell,
        },
        model.coupling_scale,
        stride_minutes,
    )
    total = balancer.combine(losses)
    total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    report = {"total": float(total.detach())}
    for name in LOSS_NAMES:
        report[name] = float(losses[name].detach())
    for name, weight in zip(LOSS_NAMES, balancer.lambdas):
        report[f"w_{name}"] = float(weight)
    return report


def fit(
    model: STPCUQGKAN,
    batch: StormBatch,
    steps: int,
    lr: float = 1e-3,
    seed: int = 0,
) -> list[dict[str, float]]:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    balancer = ReLoBRaLo()
    history = []
    for _ in range(steps):
        history.append(training_step(model, balancer, batch, optimizer))
    return history
