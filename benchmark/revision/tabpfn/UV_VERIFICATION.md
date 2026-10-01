# uv verification — 23 September 2026

Verified on node 4 (NVIDIA A10) using a newly created `.venv` from `uv sync
--locked`, Python 3.12.3 and the committed dependency lock. The pinned upstream
TabPFN Git source was fetched and built by uv. No old virtual environment or
editable source checkout was reused.

- 57 existing feature, protocol, training-history and benchmark-audit tests passed.
- Frozen and fine-tuned grouped fold 0, seed 0 fitted states each reproduced all
  476 validation predictions within 5.6e-17 (validation R² unchanged).
- The new CLI also reproduced both variants from CSVs containing only IDs,
  guide sequence and the 100 precomputed features: 476 predictions per variant,
  maximum difference 0.0 from the stored CSV. No labels or assay metadata were
  provided to this input path.
- Sealed-partition inference/evaluation CLI was exercised for the fine-tuned
  fold 0, seed 0 state; model and input checksums were verified.
- No model retraining, held-out rescoring, feature regeneration or replacement
  of sealed benchmark predictions occurred during these packaging checks.

Cross-device floating-point sensitivity remains present. Seed 1 states trained
on a different worker and reloaded on node 4 had maximum prediction deviations
0.000583 (frozen) and 0.000739 (fine-tuned); validation R² changes were +0.0000491
and −0.0001845, respectively. These two checks **do not pass** strict
`rtol=1e-5, atol=1e-6` prediction equality, consistent with the previously
recorded portability audit. This is not concealed as universal exact parity.
All sealed original predictions and fit artifacts remain unchanged.

Machine-readable evidence is in `reload-verification.json` and
`input-verification.json`. Canonical logs and comparison CSVs are under
`evaluation/tabpfn-benchmark-v1/{uv-portability-20260923,uv-input-20260923,uv-cli-20260923}`
on node 4 SCRATCH. The package does not claim verification of a second full
60-fit training matrix, unseen-input feature generation or other hardware.
