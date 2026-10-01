# Revised benchmark code

This directory contains the dataset reconstruction, model controls and evaluation
code for the completed revision archived at
[Zenodo version 2](https://doi.org/10.5281/zenodo.23001225).

Use the repository [reproduction guide](../../REPRODUCING.md) for portable
commands. The dataset has 4,098 records, five target/related-guide grouped folds,
five unselected random partitions and the full 1,047-record HeLa cohort.

## Main entry points

- `audit_data.py`, `correct_takayuki.py`, `regenerate_features.py`,
  `audit_provenance.py`, `map_targets.py` and `make_splits.py` implement the
  dataset/provenance pipeline. `../../scripts/reproduce_dataset.py` runs the
  reproducible stages using frozen correction and target audits.
- `baselines.py` contains training mean, guide/thermodynamic/combined ridge,
  Ui-Tei functionality and fixed/calibrated i-Score references. i-Score has
  historical training overlap and is excluded from primary rankings.
- `evaluate_predictions.py`, `prediction_artifacts.py` and
  `summarize_sensitivities.py` validate complete prediction matrices and compute
  shared metrics, conditional intervals and paired controls.
- `reference_models/` contains all five selected Agentomics pipelines and models.
- `reference_results/code/` adds the fold-specific reference predictions using the
  frozen analysis implementation.
- `tabpfn/` contains the frozen/fine-tuned TabPFN benchmark and locked runtime.
- `tests/` contains CPU regressions and separate real-runtime checks.

## Preserved experiment provenance

The model audit Markdown files are dated working audit logs. Their intermediate
status statements describe their respective dates, not outstanding revision
work. The final experiment includes 180 primary competitor fits, 48 sensitivity
fits, 30 frozen TabPFN context fits, 30 TabPFN fine-tuning fits and five selected
Agentomics pipelines. There were 95 completed Agentomics search iterations.

The `run_*matrix.py`, `control_node6_sensitivities.py`, `run_node6_sensitivity.py`
and TabPFN cluster launchers preserve the original multi-node orchestration.
They include site-specific paths and worker assumptions. They are provenance,
not the clean-machine entry point. Portable single-run commands use the same
model wrappers and frozen inputs without requiring access to those SSH nodes.

Input-extent and schedule controls are separate experiments. In particular,
`--original` on the legacy shell wrapper is not a universal substitute for the
reviewed sensitivity scripts. Consult the method audits and
`sensitivity_comparisons.json` for each paired control.
