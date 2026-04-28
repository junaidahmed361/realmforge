from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from app.features.window_builder import build_windows_for_stay


def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten_window_rows(rows):
    out = []
    for r in rows:
        row = {
            "subject_id": r.subject_id,
            "hadm_id": r.hadm_id,
            "window_start": r.window_start,
            "window_end": r.window_end,
        }
        row.update(r.features)
        row.update(r.actions)
        row.update(r.outcomes)
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    proc = Path(cfg["paths"]["processed_root"])
    size_h = int(cfg["time_windows"]["size_hours"])

    cohort = pd.read_parquet(proc / "hf_cohort.parquet")

    # Expected prejoined events file built from MIMIC modules.
    # Columns: subject_id, hadm_id, charttime, event_type, name, valuenum, action cols...
    events_path = proc / "hf_events.parquet"
    if not events_path.exists():
        raise FileNotFoundError(
            f"Missing {events_path}. Build event extraction first from "
            "labevents, chartevents, inputevents, outputevents, procedures."
        )
    events = pd.read_parquet(events_path)

    all_rows = []
    for r in cohort.itertuples(index=False):
        stay = events[(events.subject_id == r.subject_id) & (events.hadm_id == r.hadm_id)]
        rows = build_windows_for_stay(
            stay,
            subject_id=r.subject_id,
            hadm_id=r.hadm_id,
            admittime=r.admittime,
            dischtime=r.dischtime,
            size_hours=size_h,
        )
        all_rows.extend(rows)

    df = _flatten_window_rows(all_rows)
    out = proc / f"hf_windows_{size_h}h.parquet"
    df.to_parquet(out, index=False)
    print(f"[hf_ebwm] windows rows={len(df)} saved={out}")


if __name__ == "__main__":
    main()
