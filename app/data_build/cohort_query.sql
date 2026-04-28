-- Heart failure cohort (MIMIC-IV) by ICD codes and LOS >= 24h
WITH dx AS (
  SELECT DISTINCT d.subject_id, d.hadm_id
  FROM diagnoses_icd d
  WHERE
    (d.icd_version = 9 AND d.icd_code LIKE '428%')
    OR
    (d.icd_version = 10 AND d.icd_code LIKE 'I50%')
), base AS (
  SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    p.anchor_age AS age,
    EXTRACT(EPOCH FROM (a.dischtime - a.admittime)) / 3600.0 AS los_hours,
    a.hospital_expire_flag
  FROM admissions a
  JOIN patients p USING (subject_id)
  JOIN dx USING (subject_id, hadm_id)
)
SELECT *
FROM base
WHERE age >= 18 AND los_hours >= 24
