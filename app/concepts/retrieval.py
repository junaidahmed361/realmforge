from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

HF_CONCEPTS = {
    "acute decompensated heart failure": [
        "a_iv_loop_diuretic_bucket",
        "vital_rr_latest",
        "vital_spo2_latest",
    ],
    "volume overload": ["fluid_balance", "urine_output", "vital_rr_latest"],
    "pulmonary congestion": ["vital_spo2_latest", "vital_rr_latest", "a_oxygen_escalation"],
    "cardiorenal syndrome": ["lab_creatinine_trend", "lab_bun_trend", "fluid_balance"],
    "diuretic resistance": ["a_iv_loop_diuretic_bucket", "urine_output", "lab_creatinine_trend"],
    "discharge readiness": ["a_discharge_action", "vital_hr_latest", "lab_creatinine_latest"],
    "high readmission risk": ["y_readmit30_proxy", "lab_sodium_latest", "frailty_proxy"],
}


@dataclass
class ConceptRetrievalResult:
    concept: str
    weak_labels: list[str]
    nn_indices: np.ndarray


class ConceptRetriever:
    def __init__(self, latent_matrix: np.ndarray):
        self.latent_matrix = latent_matrix
        self.nn = NearestNeighbors(n_neighbors=16, metric="cosine").fit(latent_matrix)

    def retrieve(self, concept: str, concept_embedding: np.ndarray) -> ConceptRetrievalResult:
        dist, idx = self.nn.kneighbors(concept_embedding.reshape(1, -1), return_distance=True)
        weak = HF_CONCEPTS.get(concept.lower(), [])
        return ConceptRetrievalResult(concept=concept, weak_labels=weak, nn_indices=idx[0])
