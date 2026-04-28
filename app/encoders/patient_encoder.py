from __future__ import annotations

import torch
import torch.nn as nn


class PatientEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        h = self.norm(h)
        z = self.to_latent(h[:, -1, :])
        return z
