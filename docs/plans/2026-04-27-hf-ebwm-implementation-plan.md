# HF-EBWM Implementation Plan

> For Hermes: execute in staged modules and keep safety framing explicit in every interface.

Goal: Build an end-to-end reproducible HF world-model pipeline on MIMIC-IV with latent rollouts and energy plausibility.

Architecture:
- Data layer: cohort, event extraction, windowed states.
- Modeling layer: encoder + JEPA + action-transition + energy graph + outcomes.
- Simulation/serving layer: concept retrieval + candidate trajectory rollouts + educational harness.

Tech stack: Python, PyTorch, DuckDB, FastAPI, pytest.

Tasks:
1. Cohort extraction and split by subject_id.
2. Event schema harmonization for labs/vitals/meds/procedures/io/actions.
3. Window builder with missingness masks and action/outcome labels.
4. Encoder training (z_t).
5. JEPA future predictor training.
6. Action-conditioned transition training.
7. Energy factor graph + corruption training.
8. Outcome heads and calibration.
9. Concept retrieval index and nearest-neighbor mapping.
10. Rollout simulator + utility-logit sampler.
11. Harness generator for educational cases.
12. Serving queue + continuous batching benchmark.
13. Evaluation + ablations + model card + limitations report.
