from __future__ import annotations

import torch
import torch.nn as nn


class OutcomeHeads(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.mortality = nn.Linear(latent_dim, 1)
        self.readmit30 = nn.Linear(latent_dim, 1)
        self.wrf = nn.Linear(latent_dim, 1)
        self.icu_transfer = nn.Linear(latent_dim, 1)
        self.discharge24h = nn.Linear(latent_dim, 1)
        self.los_bucket = nn.Linear(latent_dim, 4)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "y_mortality_logit": self.mortality(z).squeeze(-1),
            "y_readmit30_logit": self.readmit30(z).squeeze(-1),
            "y_wrf_logit": self.wrf(z).squeeze(-1),
            "y_icu_transfer_logit": self.icu_transfer(z).squeeze(-1),
            "y_discharge24h_logit": self.discharge24h(z).squeeze(-1),
            "y_los_bucket_logit": self.los_bucket(z),
        }
