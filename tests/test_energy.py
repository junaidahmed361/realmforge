import torch

from energy_graph.model import EnergyFactorGraph, margin_plausibility_loss


def test_energy_margin_loss_prefers_lower_positive_energy():
    e_pos = torch.tensor([0.1, 0.2, 0.3])
    e_neg = torch.tensor([1.2, 1.4, 1.8])
    loss = margin_plausibility_loss(e_pos, e_neg, margin=1.0)
    assert loss >= 0


def test_energy_forward_shape():
    model = EnergyFactorGraph(obs_dim=64, latent_dim=128)
    obs = torch.randn(16, 64)
    e = model(obs)
    assert e.shape == (16,)
