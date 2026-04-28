from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class WindowRow:
    subject_id: int
    hadm_id: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    features: dict[str, Any]
    actions: dict[str, float]
    outcomes: dict[str, float]


ACTION_COLUMNS = [
    "a_iv_loop_diuretic_bucket",
    "a_oxygen_escalation",
    "a_vasopressor_start",
    "a_beta_blocker_toggle",
    "a_acei_arb_arni_exposure",
    "a_icu_transfer",
    "a_discharge_action",
]


def latest_with_trend(
    df: pd.DataFrame, key: str, value_col: str = "valuenum"
) -> tuple[float, float, int]:
    if df.empty:
        return np.nan, 0.0, 1
    s = df.sort_values("charttime")[value_col].astype(float)
    latest = float(s.iloc[-1])
    trend = float(s.iloc[-1] - s.iloc[0]) if len(s) > 1 else 0.0
    return latest, trend, 0


def build_windows_for_stay(
    events: pd.DataFrame,
    subject_id: int,
    hadm_id: int,
    admittime: pd.Timestamp,
    dischtime: pd.Timestamp,
    size_hours: int = 6,
) -> list[WindowRow]:
    rows: list[WindowRow] = []
    cursor = pd.Timestamp(admittime)
    end = pd.Timestamp(dischtime)
    step = pd.Timedelta(hours=size_hours)

    while cursor < end:
        w_end = min(cursor + step, end)
        w = events[(events.charttime >= cursor) & (events.charttime < w_end)]

        labs = w[w.event_type == "lab"]
        vitals = w[w.event_type == "vital"]

        cr_latest, cr_trend, cr_miss = latest_with_trend(
            labs[labs.name == "creatinine"], "creatinine"
        )
        na_latest, na_trend, na_miss = latest_with_trend(labs[labs.name == "sodium"], "sodium")
        k_latest, k_trend, k_miss = latest_with_trend(labs[labs.name == "potassium"], "potassium")
        bun_latest, bun_trend, bun_miss = latest_with_trend(labs[labs.name == "bun"], "bun")

        hr_latest, hr_trend, hr_miss = latest_with_trend(vitals[vitals.name == "hr"], "hr")
        bp_latest, bp_trend, bp_miss = latest_with_trend(vitals[vitals.name == "sbp"], "sbp")
        rr_latest, rr_trend, rr_miss = latest_with_trend(vitals[vitals.name == "rr"], "rr")
        spo2_latest, spo2_trend, spo2_miss = latest_with_trend(
            vitals[vitals.name == "spo2"], "spo2"
        )

        uo = (
            float(w[w.name == "urine_output"]["valuenum"].sum())
            if not w[w.name == "urine_output"].empty
            else 0.0
        )
        fluid_in = (
            float(w[w.name == "fluid_in"]["valuenum"].sum())
            if not w[w.name == "fluid_in"].empty
            else 0.0
        )
        fluid_balance = fluid_in - uo

        actions = {
            c: float(w[c].max()) if c in w.columns and not w.empty else 0.0 for c in ACTION_COLUMNS
        }

        outcomes = {
            "y_icu_transfer": float(actions["a_icu_transfer"] > 0),
            "y_discharge_24h": float((end - w_end).total_seconds() <= 24 * 3600),
        }

        features = {
            "lab_creatinine_latest": cr_latest,
            "lab_creatinine_trend": cr_trend,
            "lab_creatinine_missing": cr_miss,
            "lab_sodium_latest": na_latest,
            "lab_sodium_trend": na_trend,
            "lab_sodium_missing": na_miss,
            "lab_potassium_latest": k_latest,
            "lab_potassium_trend": k_trend,
            "lab_potassium_missing": k_miss,
            "lab_bun_latest": bun_latest,
            "lab_bun_trend": bun_trend,
            "lab_bun_missing": bun_miss,
            "vital_hr_latest": hr_latest,
            "vital_hr_trend": hr_trend,
            "vital_hr_missing": hr_miss,
            "vital_sbp_latest": bp_latest,
            "vital_sbp_trend": bp_trend,
            "vital_sbp_missing": bp_miss,
            "vital_rr_latest": rr_latest,
            "vital_rr_trend": rr_trend,
            "vital_rr_missing": rr_miss,
            "vital_spo2_latest": spo2_latest,
            "vital_spo2_trend": spo2_trend,
            "vital_spo2_missing": spo2_miss,
            "urine_output": uo,
            "fluid_balance": fluid_balance,
        }

        rows.append(
            WindowRow(
                subject_id=int(subject_id),
                hadm_id=int(hadm_id),
                window_start=cursor,
                window_end=w_end,
                features=features,
                actions=actions,
                outcomes=outcomes,
            )
        )
        cursor = w_end

    return rows
