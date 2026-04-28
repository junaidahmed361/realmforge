from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from energy_graph.model import EnergyFactorGraph, margin_plausibility_loss


def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = _load_cfg(args.config)

    obs_dim = 64
    model = EnergyFactorGraph(obs_dim=obs_dim, latent_dim=int(cfg["model"]["latent_dim"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))

    for step in range(500):
        obs_pos = torch.randn(256, obs_dim)
        obs_neg = obs_pos[torch.randperm(obs_pos.size(0))]

        e_pos = model(obs_pos)
        e_neg = model(obs_neg)
        loss = margin_plausibility_loss(e_pos, e_neg, margin=1.0)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"step={step} energy_margin_loss={float(loss.item()):.4f}")

    out_dir = Path(cfg["paths"]["artifacts_root"]) / "energy"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "obs_dim": obs_dim}, out_dir / "energy_graph.pt")
    print(f"saved={out_dir / 'energy_graph.pt'}")


if __name__ == "__main__":
    main()
