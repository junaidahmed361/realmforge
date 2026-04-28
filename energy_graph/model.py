from __future__ import annotations

import torch
import torch.nn as nn


class ExpertPriors:
    @staticmethod
    def renal_stress(creatinine_trend: torch.Tensor, bun_trend: torch.Tensor) -> torch.Tensor:
        return torch.relu(creatinine_trend) + 0.5 * torch.relu(bun_trend)

    @staticmethod
    def hypox_congestion(
        spo2: torch.Tensor, rr: torch.Tensor, oxygen_escalation: torch.Tensor
    ) -> torch.Tensor:
        return (
            torch.relu((92.0 - spo2) / 10.0)
            + torch.relu((rr - 22.0) / 10.0)
            + 0.5 * oxygen_escalation
        )


class EnergyFactorGraph(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.obs_to_latent = nn.Linear(obs_dim, latent_dim)

        self.f_volume = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.f_congestion = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.f_renal = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.f_discharge = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.f_guideline = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, obs: torch.Tensor, expert_terms: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        z = self.obs_to_latent(obs)
        e = (
            self.f_volume(z)
            + self.f_congestion(z)
            + self.f_renal(z)
            + self.f_discharge(z)
            + self.f_guideline(z)
        )
        if expert_terms:
            for v in expert_terms.values():
                e = e + v.unsqueeze(-1)
        return e.squeeze(-1)


def margin_plausibility_loss(
    e_pos: torch.Tensor, e_neg: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    return torch.relu(margin + e_pos - e_neg).mean()
