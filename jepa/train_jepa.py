from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from encoders.patient_encoder import PatientEncoder
from jepa.losses import contrastive_jepa_loss
from jepa.model import JEPAFuturePredictor


def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = _load_cfg(args.config)

    artifacts = Path(cfg["paths"]["artifacts_root"])
    ckpt = torch.load(artifacts / "encoder" / "patient_encoder.pt", map_location="cpu")
    feature_cols = ckpt["feature_cols"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = PatientEncoder(
        input_dim=len(feature_cols),
        latent_dim=int(cfg["model"]["latent_dim"]),
        hidden_dim=int(cfg["model"]["encoder_hidden"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_layers=int(cfg["model"]["n_layers"]),
    ).to(device)
    encoder.load_state_dict(ckpt["model"])
    encoder.eval()

    predictor = JEPAFuturePredictor(latent_dim=int(cfg["model"]["latent_dim"])).to(device)
    opt = torch.optim.AdamW(predictor.parameters(), lr=float(cfg["training"]["lr"]))

    for step in range(200):
        z_t = torch.randn(256, int(cfg["model"]["latent_dim"]), device=device)
        delta_t = torch.randint(1, 5, (256,), device=device).float()
        z_future = z_t + 0.05 * torch.randn_like(z_t)

        z_pred = predictor(z_t, delta_t)
        loss = contrastive_jepa_loss(z_pred, z_future)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"step={step} loss={float(loss.item()):.4f}")

    out_dir = artifacts / "jepa"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": predictor.state_dict()}, out_dir / "jepa_future.pt")
    print(f"saved={out_dir / 'jepa_future.pt'}")


if __name__ == "__main__":
    main()
