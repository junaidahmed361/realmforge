from __future__ import annotations

import torch
import torch.nn as nn


class GenericOutcomeHeads(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        binary_heads: list[str] | None = None,
        multiclass_heads: dict[str, int] | None = None,
    ):
        super().__init__()
        self.binary_heads = nn.ModuleDict(
            {h: nn.Linear(latent_dim, 1) for h in (binary_heads or [])}
        )
        self.multiclass_heads = nn.ModuleDict(
            {h: nn.Linear(latent_dim, c) for h, c in (multiclass_heads or {}).items()}
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {f"{k}_logit": v(z).squeeze(-1) for k, v in self.binary_heads.items()}
        out.update({f"{k}_logits": v(z) for k, v in self.multiclass_heads.items()})
        return out
