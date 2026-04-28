# HF-EBWM Model Card

Intended use:
- Retrospective research and medical education simulation in heart failure trajectories.

Not intended use:
- Real-time clinical decision support.
- Treatment recommendation.

Data:
- MIMIC-IV structured EHR (optional MIMIC-IV-Note embeddings).
- Adult HF cohort using ICD-9 428* and ICD-10 I50* with LOS >= 24h.

Model components:
- Patient encoder -> latent z_t.
- JEPA future predictor z_t + delta_t -> z_t+k.
- Action-conditioned transition z_t + a_t:t+k -> z_t+k.
- Energy-based factor graph with physiologic plausibility constraints.
- Outcome heads for mortality/readmission/renal transfer/discharge/LOS.

Limitations:
- Observational confounding in action effects unless causal adjustment is added.
- Concept retrieval quality depends on weak labels and embedding quality.
- Missingness and coding variability may degrade generalization.

Safety:
- Educational candidate trajectory selection only.
- All outputs must include non-clinical-use disclaimer.
