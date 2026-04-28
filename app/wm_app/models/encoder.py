from __future__ import annotations

import torch
import torch.nn as nn


class GenericTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        return self.net(x[:, -1, :])
