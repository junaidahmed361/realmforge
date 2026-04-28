from __future__ import annotations

import time

import torch

from energy_graph.model import EnergyFactorGraph
from rollout.simulator import WorldModelSimulator
from transition.model import ActionConditionedTransition
from transition.outcome_heads import OutcomeHeads


def run(mode: str, horizon: int, n_samples: int, batch: int = 32) -> dict:
    latent_dim = 128
    transition = ActionConditionedTransition(latent_dim=latent_dim, action_dim=7)
    energy = EnergyFactorGraph(obs_dim=64, latent_dim=latent_dim)
    outcomes = OutcomeHeads(latent_dim=latent_dim)
    sim = WorldModelSimulator(transition, energy, outcomes)

    z0 = torch.randn(batch, latent_dim)
    actions = torch.randint(0, 2, (batch, n_samples, horizon, 7)).float()

    t0 = time.time()
    _ = sim.simulate_concept(z0, actions, horizon=horizon, n_samples=n_samples)
    dt = time.time() - t0

    trajectories = batch * n_samples
    return {
        "mode": mode,
        "horizon": horizon,
        "n_samples": n_samples,
        "batch": batch,
        "latency_sec": dt,
        "simulations_per_sec": batch / max(dt, 1e-9),
        "trajectories_per_sec": trajectories / max(dt, 1e-9),
        "gpu_mem_gb": float(torch.cuda.memory_allocated() / 1e9)
        if torch.cuda.is_available()
        else 0.0,
    }


if __name__ == "__main__":
    for mode in ["A_LLM_only_baseline_stub", "B_world_model_plus_llm", "C_world_model_only"]:
        for h in [4, 8, 12]:
            for s in [32, 64, 128]:
                print(run(mode, h, s))
