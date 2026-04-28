from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd
import yaml


def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _register_table(con: duckdb.DuckDBPyConnection, name: str, root: Path) -> None:
    parquet_glob = root / "**" / f"{name}.parquet"
    csv_glob = root / "**" / f"{name}.csv*"
    parquet_matches = list(root.glob(f"**/{name}.parquet"))
    csv_matches = list(root.glob(f"**/{name}.csv")) + list(root.glob(f"**/{name}.csv.gz"))

    if parquet_matches:
        con.execute(
            f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{parquet_glob}')"
        )
        return
    if csv_matches:
        con.execute(
            "CREATE OR REPLACE VIEW "
            f"{name} AS SELECT * FROM read_csv_auto('{csv_glob}', HEADER=TRUE)"
        )
        return
    raise FileNotFoundError(f"Could not find {name}.parquet/.csv(.gz) under {root}")


def subject_split(df: pd.DataFrame, train: float, val: float, seed: int = 42) -> pd.DataFrame:
    subj = pd.Series(df["subject_id"].unique()).sample(frac=1.0, random_state=seed).tolist()
    n = len(subj)
    n_train = int(n * train)
    n_val = int(n * val)
    train_set = set(subj[:n_train])
    val_set = set(subj[n_train : n_train + n_val])

    def _split(sid: int) -> str:
        if sid in train_set:
            return "train"
        if sid in val_set:
            return "val"
        return "test"

    out = df.copy()
    out["split"] = out["subject_id"].map(_split)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    raw_root = Path(cfg["paths"]["raw_root"])
    out_root = Path(cfg["paths"]["processed_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    for t in ["patients", "admissions", "diagnoses_icd"]:
        _register_table(con, t, raw_root)

    query_path = Path(__file__).with_name("cohort_query.sql")
    cohort = con.execute(query_path.read_text(encoding="utf-8")).df()

    split_cfg = cfg["cohort"]["split"]
    cohort = subject_split(
        cohort,
        train=float(split_cfg["train"]),
        val=float(split_cfg["val"]),
        seed=int(cfg["project"]["seed"]),
    )
    out_path = out_root / "hf_cohort.parquet"
    cohort.to_parquet(out_path, index=False)
    print(f"[hf_ebwm] cohort rows={len(cohort)} saved={out_path}")


if __name__ == "__main__":
    main()
