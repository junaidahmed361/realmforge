# Limitations and Risk Notes

1) Confounding by indication
Medication/intervention effects are observational and can reflect severity bias.

2) Label leakage
Strict timestamp filtering is required. No post-discharge artifacts before discharge-time prediction.

3) Transportability
MIMIC is single-center ICU-heavy data; external validity may be limited.

4) Missingness and measurement bias
Sparse labs/vitals are informative but can induce model shortcuts.

5) Causal claims
HF-EBWM currently supports plausibility-constrained simulation, not causal treatment effect estimation.
