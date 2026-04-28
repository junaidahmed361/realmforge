from __future__ import annotations

import torch

from app.energy_graph.model import EnergyFactorGraph
from app.rollout.simulator import WorldModelSimulator
from app.transition.model import ActionConditionedTransition
from app.transition.outcome_heads import OutcomeHeads


def main() -> None:
    torch.manual_seed(42)

    latent_dim = 128
    action_dim = 7
    obs_dim = 64

    transition = ActionConditionedTransition(latent_dim=latent_dim, action_dim=action_dim)
    energy = EnergyFactorGraph(obs_dim=obs_dim, latent_dim=latent_dim)
    outcomes = OutcomeHeads(latent_dim=latent_dim)
    sim = WorldModelSimulator(transition, energy, outcomes)

    batch = 1
    n_samples = 12
    horizon = 6

    z0 = torch.randn(batch, latent_dim)
    actions = torch.randint(0, 2, (batch, n_samples, horizon, action_dim)).float()

    out = sim.simulate_concept(
        z0=z0,
        candidate_action_sequences=actions,
        horizon=horizon,
        n_samples=n_samples,
    )

    utility = out.utility_logits.flatten()
    topk = torch.topk(utility, k=3)

    print("RealmForge Hello World")
    print(f"latent shape: {tuple(out.latent.shape)}")
    print(f"utility logits shape: {tuple(out.utility_logits.shape)}")
    print(f"top-3 scenario indices: {topk.indices.tolist()}")
    print(f"top-3 utility logits: {[round(float(v), 4) for v in topk.values]}")


if __name__ == "__main__":
    main()
