from __future__ import annotations

import torch
import torch.nn as nn


class ActionConditionedTransition(nn.Module):
    def __init__(self, latent_dim: int = 128, action_dim: int = 7, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self, z_t: torch.Tensor, a_seq_summary: torch.Tensor, delta_t: torch.Tensor
    ) -> torch.Tensor:
        if delta_t.ndim == 1:
            delta_t = delta_t.unsqueeze(-1)
        x = torch.cat([z_t, a_seq_summary, delta_t], dim=-1)
        return self.net(x)
