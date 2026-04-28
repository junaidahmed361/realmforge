from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EventToken:
    token: str
    charttime: pd.Timestamp
    value: float | None


def bucketize_numeric(x: float, lo: float, hi: float, n_bins: int = 32) -> int:
    x = float(np.clip(x, lo, hi))
    return int(np.floor((x - lo) / max((hi - lo), 1e-9) * (n_bins - 1)))


def tokenize_diagnoses(dx_df: pd.DataFrame) -> list[EventToken]:
    out: list[EventToken] = []
    for r in dx_df.itertuples(index=False):
        tok = f"dx:{r.icd_version}:{r.icd_code}"
        out.append(EventToken(tok, pd.Timestamp(r.charttime), None))
    return out


def tokenize_labs(labs_df: pd.DataFrame, n_bins: int = 32) -> list[EventToken]:
    out: list[EventToken] = []
    for r in labs_df.itertuples(index=False):
        b = bucketize_numeric(r.valuenum, lo=float(r.ref_low), hi=float(r.ref_high), n_bins=n_bins)
        tok = f"lab:{r.itemid}:b{b}"
        out.append(EventToken(tok, pd.Timestamp(r.charttime), float(r.valuenum)))
    return out


def tokenize_vitals(vitals_df: pd.DataFrame, n_bins: int = 32) -> list[EventToken]:
    out: list[EventToken] = []
    for r in vitals_df.itertuples(index=False):
        lo, hi = float(r.min_range), float(r.max_range)
        b = bucketize_numeric(r.valuenum, lo=lo, hi=hi, n_bins=n_bins)
        tok = f"vital:{r.itemid}:b{b}"
        out.append(EventToken(tok, pd.Timestamp(r.charttime), float(r.valuenum)))
    return out


def add_gap_tokens(tokens: Iterable[EventToken], max_gap_hours: int = 48) -> list[EventToken]:
    seq = sorted(tokens, key=lambda x: x.charttime)
    if not seq:
        return []
    out = [seq[0]]
    for prev, cur in zip(seq, seq[1:], strict=False):
        gap_h = (cur.charttime - prev.charttime).total_seconds() / 3600
        if gap_h >= 1:
            gap_h = min(gap_h, max_gap_hours)
            out.append(EventToken(token=f"gap:{int(gap_h)}h", charttime=cur.charttime, value=gap_h))
        out.append(cur)
    return out
