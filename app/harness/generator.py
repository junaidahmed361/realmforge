from __future__ import annotations

from typing import Any


def generate_training_harness(
    case_id: str,
    patient_background: dict[str, Any],
    initial_state: dict[str, Any],
    learner_objective: str,
    available_actions: list[str],
    simulated_timeline: list[dict[str, Any]],
    expected_changes: list[str],
    scoring_rubric: dict[str, float],
    failure_modes: list[str],
    debrief: str,
) -> dict[str, Any]:
    return {
        "safety_notice": (
            "Research and education only. "
            "Candidate trajectory selection, not treatment recommendation."
        ),
        "case_opening": f"HF-EBWM Case {case_id}",
        "patient_background": patient_background,
        "initial_vitals": initial_state.get("vitals", {}),
        "initial_labs": initial_state.get("labs", {}),
        "learner_objective": learner_objective,
        "available_actions": available_actions,
        "simulated_timeline": simulated_timeline,
        "expected_changes": expected_changes,
        "scoring_rubric": scoring_rubric,
        "failure_modes": failure_modes,
        "debrief": debrief,
    }
