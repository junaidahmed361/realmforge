from __future__ import annotations

import torch

from app.energy_graph.model import EnergyFactorGraph
from app.rollout.simulator import WorldModelSimulator
from app.transition.model import ActionConditionedTransition
from app.transition.outcome_heads import OutcomeHeads


def main() -> None:
    torch.manual_seed(7)

    latent_dim = 128
    action_dim = 7
    obs_dim = 64

    transition = ActionConditionedTransition(latent_dim=latent_dim, action_dim=action_dim)
    energy = EnergyFactorGraph(obs_dim=obs_dim, latent_dim=latent_dim)
    outcomes = OutcomeHeads(latent_dim=latent_dim)
    sim = WorldModelSimulator(transition, energy, outcomes)

    # One warehouse context, 3 candidate replenishment policies, 5-day horizon.
    batch = 1
    n_samples = 3
    horizon = 5

    policy_names = [
        "no_replenishment",
        "small_daily_restock",
        "medium_burst_restock",
    ]

    z0 = torch.randn(batch, latent_dim)

    # action vector semantics here are illustrative only.
    actions = torch.zeros(batch, n_samples, horizon, action_dim)

    # Policy 0: no replenishment.
    # keep zeros

    # Policy 1: small restock daily (feature 0 low, feature 1 moderate)
    actions[:, 1, :, 0] = 1.0
    actions[:, 1, :, 1] = 0.5

    # Policy 2: medium burst upfront then taper.
    actions[:, 2, 0:2, 0] = 2.0
    actions[:, 2, 0:2, 1] = 1.0
    actions[:, 2, 2:, 0] = 0.5

    out = sim.simulate_concept(
        z0=z0,
        candidate_action_sequences=actions,
        horizon=horizon,
        n_samples=n_samples,
    )

    utility = out.utility_logits.view(batch, n_samples)[0]
    best_idx = int(torch.argmax(utility).item())

    print("Supply Chain Mini Realm")
    for i, name in enumerate(policy_names):
        print(f"{name}: utility={float(utility[i]):.4f}")
    print(f"best_policy={policy_names[best_idx]}")


if __name__ == "__main__":
    main()
