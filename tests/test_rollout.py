import torch

from app.energy_graph.model import EnergyFactorGraph
from app.rollout.simulator import WorldModelSimulator
from app.transition.model import ActionConditionedTransition
from app.transition.outcome_heads import OutcomeHeads


def test_rollout_shapes():
    transition = ActionConditionedTransition(latent_dim=128, action_dim=7)
    energy = EnergyFactorGraph(obs_dim=64, latent_dim=128)
    outcomes = OutcomeHeads(latent_dim=128)
    sim = WorldModelSimulator(transition, energy, outcomes)

    z0 = torch.randn(2, 128)
    actions = torch.randint(0, 2, (2, 4, 6, 7)).float()
    out = sim.simulate_concept(z0, actions, horizon=6, n_samples=4)

    assert out.latent.shape[0] == 8
    assert out.utility_logits.shape[0] == 8
