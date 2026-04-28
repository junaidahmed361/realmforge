from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from app.encoders.dataset import HFWindowSequenceDataset
from app.encoders.patient_encoder import PatientEncoder


def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = _load_cfg(args.config)

    proc = Path(cfg["paths"]["processed_root"])
    size_h = int(cfg["time_windows"]["size_hours"])
    df = pd.read_parquet(proc / f"hf_windows_{size_h}h.parquet")

    excluded = {"subject_id", "hadm_id", "window_start", "window_end"}
    feature_cols = [c for c in df.columns if c not in excluded and not c.startswith("y_")]

    train_df = df.merge(
        pd.read_parquet(proc / "hf_cohort.parquet")[["subject_id", "hadm_id", "split"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )
    train_df = train_df[train_df["split"] == "train"]

    ds = HFWindowSequenceDataset(train_df, feature_cols=feature_cols, seq_len=8)
    dl = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatientEncoder(
        input_dim=len(feature_cols),
        latent_dim=int(cfg["model"]["latent_dim"]),
        hidden_dim=int(cfg["model"]["encoder_hidden"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_layers=int(cfg["model"]["n_layers"]),
    ).to(device)

    head = nn.Linear(int(cfg["model"]["latent_dim"]), len(feature_cols)).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=float(cfg["training"]["lr"])
    )
    loss_fn = nn.MSELoss()

    model.train()
    head.train()
    for epoch in range(int(cfg["training"]["max_epochs"])):
        running = 0.0
        for b in dl:
            x = b.x.to(device)
            y = b.y.to(device)
            z = model(x)
            yhat = head(z)
            loss = loss_fn(yhat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
        print(f"epoch={epoch} loss={running / max(len(dl), 1):.4f}")

    out_dir = Path(cfg["paths"]["artifacts_root"]) / "encoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "feature_cols": feature_cols, "config": cfg},
        out_dir / "patient_encoder.pt",
    )
    print(f"saved={out_dir / 'patient_encoder.pt'}")


if __name__ == "__main__":
    main()
