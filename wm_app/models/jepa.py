from __future__ import annotations

import torch
import torch.nn as nn


class GenericJEPAPredictor(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_t: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        if delta_t.ndim == 1:
            delta_t = delta_t.unsqueeze(-1)
        return self.net(torch.cat([z_t, delta_t], dim=-1))
