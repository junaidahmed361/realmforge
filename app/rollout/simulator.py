from __future__ import annotations

from dataclasses import dataclass

import torch

from app.energy_graph.model import EnergyFactorGraph
from app.transition.model import ActionConditionedTransition
from app.transition.outcome_heads import OutcomeHeads


@dataclass
class RolloutResult:
    latent: torch.Tensor
    energies: torch.Tensor
    utility_logits: torch.Tensor
    outcome_logits: dict[str, torch.Tensor]


def utility_logit(
    protocol_alignment,
    educational_relevance,
    mortality_risk,
    readmission_risk,
    renal_injury_risk,
    physiologic_energy,
    impossible_penalty,
):
    return (
        protocol_alignment
        + educational_relevance
        - mortality_risk
        - readmission_risk
        - renal_injury_risk
        - physiologic_energy
        - impossible_penalty
    )


class WorldModelSimulator:
    def __init__(
        self,
        transition: ActionConditionedTransition,
        energy: EnergyFactorGraph,
        outcomes: OutcomeHeads,
    ):
        self.transition = transition
        self.energy = energy
        self.outcomes = outcomes

    @torch.no_grad()
    def simulate_concept(
        self,
        z0: torch.Tensor,
        candidate_action_sequences: torch.Tensor,
        horizon: int,
        n_samples: int = 64,
    ):
        # z0: [B, D], actions: [B, S, H, A]
        B, S, H, A = candidate_action_sequences.shape
        D = z0.shape[-1]
        z = z0.unsqueeze(1).repeat(1, S, 1).reshape(B * S, D)

        latents = [z]
        energies = []

        for t in range(min(horizon, H)):
            a_t = candidate_action_sequences[:, :, t, :].reshape(B * S, A)
            dt = torch.ones(B * S, device=z.device)
            z = self.transition(z, a_t, dt)
            e_t = self.energy(torch.cat([z, z[:, : min(64, z.shape[1])]], dim=-1)[:, :64])
            latents.append(z)
            energies.append(e_t)

        latent_traj = torch.stack(latents, dim=1)
        energy_traj = (
            torch.stack(energies, dim=1) if energies else torch.zeros(B * S, 1, device=z.device)
        )

        final_z = latent_traj[:, -1, :]
        y = self.outcomes(final_z)

        mort = torch.sigmoid(y["y_mortality_logit"])
        readm = torch.sigmoid(y["y_readmit30_logit"])
        renal = torch.sigmoid(y["y_wrf_logit"])

        protocol_alignment = 0.5 * torch.ones_like(mort)
        educational_relevance = 0.5 * torch.ones_like(mort)
        impossible = torch.relu(energy_traj.mean(dim=1) - 10.0)
        util = utility_logit(
            protocol_alignment,
            educational_relevance,
            mort,
            readm,
            renal,
            energy_traj.mean(dim=1),
            impossible,
        )

        return RolloutResult(
            latent=latent_traj, energies=energy_traj, utility_logits=util, outcome_logits=y
        )
