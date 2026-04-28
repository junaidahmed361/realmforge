from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceBatch:
    x: torch.Tensor
    y: torch.Tensor


class HFWindowSequenceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_cols: list[str], seq_len: int = 8):
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        frame = frame.sort_values(["subject_id", "hadm_id", "window_start"]).reset_index(drop=True)

        self.sequences = []
        self.targets = []
        for _, g in frame.groupby(["subject_id", "hadm_id"], sort=False):
            x = (
                g[feature_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
            if len(x) <= seq_len:
                continue
            for i in range(len(x) - seq_len):
                self.sequences.append(x[i : i + seq_len])
                self.targets.append(x[i + seq_len])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return SequenceBatch(
            x=torch.tensor(self.sequences[idx], dtype=torch.float32),
            y=torch.tensor(self.targets[idx], dtype=torch.float32),
        )
