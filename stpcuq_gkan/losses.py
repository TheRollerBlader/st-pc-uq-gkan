"""Five-task objective and ReLoBRaLo adaptive weights.

The tasks are localized magnetic perturbation, SuperMAG-style SME envelope,
geoelectric field, ordinal GIC risk, and the Newell physics residual.
ReLoBRaLo keeps a slowly updated softmax over loss ratios so the global SME
term does not silence the local dB term:

    λ_i(t) = α softmax(L_i(t) / (T L_i(t_ref))) + (1 - α) λ_i(t - 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


LOSS_NAMES = ("db", "sme", "efield", "risk", "physics")


def _mse(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).square().mean()


def task_losses(
    prediction: dict[str, Tensor],
    batch: dict[str, Tensor],
    coupling_scale: Tensor,
    stride_minutes: float = 1.0,
) -> dict[str, Tensor]:
    db = _mse(prediction["db"], batch["db"])
    sme = _mse(prediction["sme"], batch["sme"])
    efield = _mse(prediction["e_field"], batch["e_field"])
    risk = nn.functional.cross_entropy(
        prediction["risk_logits"].reshape(-1, prediction["risk_logits"].size(-1)),
        batch["risk"].reshape(-1),
    )

    d_sme = (prediction["sme"][1:, 0] - prediction["sme"][:-1, 0]) / stride_minutes
    driver = batch["newell"][1:]
    driver = driver / driver.detach().mean().clamp(min=1e-6)
    if d_sme.numel() == 0:
        physics = prediction["sme"].sum() * 0.0
    else:
        physics = (d_sme - coupling_scale * driver).square().mean()
    return {
        "db": db,
        "sme": sme,
        "efield": efield,
        "risk": risk,
        "physics": physics,
    }


@dataclass
class ReLoBRaLo:
    """Relative loss balancing with random lookback."""

    n_losses: int = 5
    mix: float = 0.5
    temperature: float = 1.0
    lookback_probability: float = 0.1
    lambdas: Tensor = field(init=False)
    initial: Tensor | None = field(default=None, init=False)
    history: list[Tensor] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.lambdas = torch.full((self.n_losses,), 1.0 / self.n_losses)

    def combine(self, losses: dict[str, Tensor]) -> Tensor:
        values = torch.stack([losses[name].detach() for name in LOSS_NAMES])
        if self.initial is None:
            self.initial = values.clamp(min=1e-8).clone()
        self.history.append(values.clamp(min=1e-8).clone())

        reference = self.initial
        if len(self.history) > 1 and torch.rand(()) < self.lookback_probability:
            index = int(torch.randint(0, len(self.history) - 1, ()).item())
            reference = self.history[index]

        ratios = values.clamp(min=1e-8) / (self.temperature * reference.clamp(min=1e-8))
        balanced = torch.softmax(ratios, dim=0)
        self.lambdas = self.mix * balanced + (1.0 - self.mix) * self.lambdas
        weights = [float(value) for value in self.lambdas.detach()]
        total = torch.zeros((), device=losses["db"].device, dtype=losses["db"].dtype)
        for weight, name in zip(weights, LOSS_NAMES):
            total = total + weight * losses[name]
        return total
