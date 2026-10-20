"""Full ST-PC-UQ-GKAN forward pass.

Solar wind is encoded by TKAN and broadcast to every station with a learned
gate. It is not written into the adjacency matrix. Station states then move
across the static geographic graph through KAN message and update functions.
Three forecast horizons are emitted together: +10, +30, and +60 minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from stpcuq_gkan.kan import RBFKANLayer
from stpcuq_gkan.tkan import TKANEncoder

HORIZONS_MINUTES = (10, 30, 60)


@dataclass
class ModelConfig:
    solar_features: int = 5
    node_features: int = 10
    hidden: int = 32
    basis: int = 8
    graph_layers: int = 1
    edge_features: int = 3
    horizons: tuple[int, ...] = HORIZONS_MINUTES


def scatter_mean(messages: Tensor, index: Tensor, num_nodes: int) -> Tensor:
    batch, edges, features = messages.shape
    output = messages.new_zeros(batch, num_nodes, features)
    counts = messages.new_zeros(batch, num_nodes, 1)
    expanded = index.view(1, edges, 1).expand(batch, edges, features)
    output.scatter_add_(1, expanded, messages)
    count_index = index.view(1, edges, 1).expand(batch, edges, 1)
    ones = torch.ones(batch, edges, 1, device=messages.device, dtype=messages.dtype)
    counts.scatter_add_(1, count_index, ones)
    return output / counts.clamp(min=1.0)


class GraphKANLayer(nn.Module):
    def __init__(self, hidden: int, edge_features: int, num_basis: int) -> None:
        super().__init__()
        self.message = RBFKANLayer(hidden * 2 + edge_features, hidden, num_basis)
        self.update = RBFKANLayer(hidden * 2, hidden, num_basis)

    def forward(self, nodes: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        source, target = edge_index
        paired = torch.cat(
            [
                nodes[:, target],
                nodes[:, source],
                edge_attr.unsqueeze(0).expand(nodes.size(0), -1, -1),
            ],
            dim=-1,
        )
        messages = self.message(paired)
        aggregated = scatter_mean(messages, target, nodes.size(1))
        return self.update(torch.cat([nodes, aggregated], dim=-1))


class STPCUQGKAN(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        hidden = self.config.hidden
        basis = self.config.basis
        horizons = len(self.config.horizons)
        self.encoder = TKANEncoder(self.config.solar_features, hidden, basis)
        self.node_in = RBFKANLayer(self.config.node_features, hidden, basis)
        self.driver_gate = nn.Linear(hidden, hidden)
        self.layers = nn.ModuleList(
            [
                GraphKANLayer(hidden, self.config.edge_features, basis)
                for _ in range(self.config.graph_layers)
            ]
        )
        self.db_head = RBFKANLayer(hidden, horizons * 2, basis)
        self.e_head = RBFKANLayer(hidden, horizons, basis)
        self.risk_head = RBFKANLayer(hidden, horizons * 3, basis)
        self.coupling_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        solar_wind: Tensor,
        nodes: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> dict[str, Tensor]:
        context = self.encoder(solar_wind)
        hidden = self.node_in(nodes)
        gate = torch.sigmoid(self.driver_gate(context)).unsqueeze(1)
        hidden = hidden + gate * context.unsqueeze(1)
        for layer in self.layers:
            hidden = hidden + layer(hidden, edge_index, edge_attr)

        horizons = len(self.config.horizons)
        db = self.db_head(hidden).reshape(hidden.size(0), hidden.size(1), horizons, 2)
        north = db[..., 0]
        sme = north.amax(dim=1) - north.amin(dim=1)
        e_field = self.e_head(hidden).reshape(hidden.size(0), hidden.size(1), horizons)
        risk_logits = self.risk_head(hidden).reshape(
            hidden.size(0), hidden.size(1), horizons, 3
        )
        return {
            "db": db,
            "sme": sme,
            "e_field": e_field,
            "risk_logits": risk_logits,
        }
