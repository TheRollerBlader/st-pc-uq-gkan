"""Gaussian radial-basis Kolmogorov-Arnold layer.

B-spline KANs are a poor fit for a deep spatiotemporal graph because the
Cox-de Boor recursion is sequential. This layer follows the FastKAN choice in
the architecture note: a smooth Gaussian basis on each edge, plus a SiLU
residual path so the spline grid is not the only route for gradient.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RBFKANLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_basis: int = 8,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        if num_basis < 2:
            raise ValueError("num_basis must be at least 2")
        self.in_features = in_features
        self.out_features = out_features
        self.num_basis = num_basis
        centers = torch.linspace(grid_range[0], grid_range[1], num_basis)
        self.register_buffer("centers", centers)
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self.coeff = nn.Parameter(torch.empty(out_features, in_features, num_basis))
        self.base = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.base.weight)
        nn.init.zeros_(self.base.bias)
        nn.init.normal_(self.coeff, std=0.02)

    def basis(self, x: Tensor) -> Tensor:
        sigma = self.log_sigma.exp().clamp(min=0.05, max=2.0)
        # tanh keeps storm spikes inside the basis grid.
        bounded = torch.tanh(x)
        diff = bounded.unsqueeze(-1) - self.centers
        return torch.exp(-0.5 * (diff / sigma).square())

    def forward(self, x: Tensor) -> Tensor:
        spline = torch.einsum("...ib,oib->...o", self.basis(x), self.coeff)
        return self.base(torch.nn.functional.silu(x)) + spline

    @torch.no_grad()
    def coefficient_importance(self) -> Tensor:
        """Mean absolute basis coefficient for each input-output edge."""
        return self.coeff.abs().mean(dim=-1)
