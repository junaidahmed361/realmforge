from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from transition.model import ActionConditionedTransition


def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = _load_cfg(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = int(cfg["model"]["latent_dim"])
    model = ActionConditionedTransition(latent_dim=latent_dim, action_dim=7).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))
    mse = nn.MSELoss()

    for step in range(500):
        z_t = torch.randn(256, latent_dim, device=device)
        a = torch.randint(0, 3, (256, 7), device=device).float()
        delta_t = torch.randint(1, 5, (256,), device=device).float()
        z_next_true = z_t + 0.03 * torch.randn_like(z_t) + 0.01 * a.mean(dim=-1, keepdim=True)

        z_next_pred = model(z_t, a, delta_t)
        loss = mse(z_next_pred, z_next_true)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"step={step} transition_loss={float(loss.item()):.4f}")

    out_dir = Path(cfg["paths"]["artifacts_root"]) / "transition"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out_dir / "transition.pt")
    print(f"saved={out_dir / 'transition.pt'}")


if __name__ == "__main__":
    main()
