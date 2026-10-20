"""KAN-gated recurrent encoder for the upstream solar-wind history.

A static KAN cannot represent the magnetosphere loading and unloading cycle.
TKAN puts a KAN inside each LSTM gate:

    f_t = sigmoid(KAN_f([h_{t-1}, x_t]))

The same replacement is used for the input gate, the output gate, and the
cell candidate. The demo and the module default to a shorter window than the
120–180 minute lookback specified for OMNI data; the cell itself is length
agnostic.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from stpcuq_gkan.kan import RBFKANLayer


class TKANCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_basis: int = 8) -> None:
        super().__init__()
        gate_in = input_size + hidden_size
        self.hidden_size = hidden_size
        self.forget = RBFKANLayer(gate_in, hidden_size, num_basis)
        self.input = RBFKANLayer(gate_in, hidden_size, num_basis)
        self.output = RBFKANLayer(gate_in, hidden_size, num_basis)
        self.cell = RBFKANLayer(gate_in, hidden_size, num_basis)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        h, c = state
        combined = torch.cat([h, x], dim=-1)
        forget = torch.sigmoid(self.forget(combined))
        ingested = torch.sigmoid(self.input(combined))
        exposed = torch.sigmoid(self.output(combined))
        candidate = torch.tanh(self.cell(combined))
        c_next = forget * c + ingested * candidate
        h_next = exposed * torch.tanh(c_next)
        return h_next, c_next


class TKANEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_basis: int = 8) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = TKANCell(input_size, hidden_size, num_basis)

    def forward(self, sequence: Tensor) -> Tensor:
        """Encode ``sequence`` of shape (batch, time, features) into (batch, hidden)."""
        batch, steps, _ = sequence.shape
        h = sequence.new_zeros(batch, self.hidden_size)
        c = sequence.new_zeros(batch, self.hidden_size)
        for index in range(steps):
            h, c = self.cell(sequence[:, index], (h, c))
        return h
