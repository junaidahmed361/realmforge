from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_jepa_loss(
    pred: torch.Tensor, true: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    true = F.normalize(true, dim=-1)
    logits = pred @ true.t() / temperature
    labels = torch.arange(pred.size(0), device=pred.device)
    return F.cross_entropy(logits, labels)
