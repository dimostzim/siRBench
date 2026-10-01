# TabPFN-3.5 with uv

The frozen and fine-tuned rows share one pinned runtime. `uv.lock` replaces the
machine-local TabPFN source path in `runtime.lock.txt` with the exact upstream
Git commit used in the benchmark. All recorded package versions are retained,
including TabPFN 9.0.0, PyTorch 2.8.0+cu126, NumPy 2.5.3, pandas 3.0.6 and
scikit-learn 1.9.1. This environment targets Linux x86-64, Python 3.12.3 and an
NVIDIA CUDA GPU; it does not imply tested CPU/macOS training compatibility.

## Install

```bash
cd siRBench/benchmark/revision/tabpfn
uv sync --locked
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export R=/absolute/path/to/benchmark-workspace
```

Retain the workspace layout documented in `README.md`. The sealed corrected
dataset, feature manifest and partitions are dependencies, not regenerated or
reselected by this package. The runner checks their hashes. Source/cell-line
columns may be present for auditing but cannot enter the allowlisted 176 model
features (76 guide one-hot + 100 guide/central-duplex features).

Obtain the official TabPFN-3.5 foundation checkpoint through your own authorized
Prior Labs access, accepting the applicable terms. Place it at
`$R/evaluation/tabpfn-v1/models/v3.5/tabpfn-v3.5-20260909.safetensors`.
The expected SHA256 is recorded in `README.md`; it is checked before training.
This code package includes neither foundation weights, fitted TabPFN artifacts,
access tokens nor cached credentials. Existing authorized fitted states remain
in the canonical benchmark workspace. Do not commit credentials or gated weights.

## Train the two variants

```bash
uv run --locked python run_benchmark.py \
  --root "$R" --axis grouped --fold 0 --seed 0 \
  --output "$R/evaluation/tabpfn-reproduction/grouped/fold_0/seed_0/attempt_1"
```

Choose a new output directory. The unchanged benchmark runner creates frozen
and fine-tuned fits together. The frozen model conditions on training examples;
the fine-tuned model additionally updates weights and selects its checkpoint
using validation MSE (equivalent to maximizing validation R²). Test/HeLa rows
are evaluated only after fitting and selection. Original settings remain eight
inference estimators, a 100-epoch cap and patience 20. The full matrix comprises
`axis={grouped,random}`, five folds and three seeds: 30 context fits plus 30
fine-tuning runs. Packaging validation did not rerun these training experiments.

## Reload, predict and evaluate a saved fit

```bash
uv run --locked python predict_saved.py \
  --root "$R" \
  --model-dir "$R/evaluation/tabpfn-benchmark-v1/runs/grouped/fold_0/seed_0/attempt_2/tabpfn35_frozen" \
  --partition grouped/fold_0/val.csv \
  --output "$R/evaluation/tabpfn-reproduction/frozen-validation" --evaluate
```

Use `tabpfn35_finetuned` for the fine-tuned state. `--partition` is an existing
key from `evaluation/protocol-v1/manifest.json` (including test or full-HeLa
cohorts). Prediction loads the selected fitted state, verifies its checksum,
resets the saved seed and makes one complete-cohort prediction call. The output
contains ordered IDs/predictions and a provenance report. `--evaluate` adds
metrics after prediction; labels never enter inference. Shared feature and
metric helpers are imported from this repository, not copied independently.

For the paper's aggregate tables, continue using the original common-analysis
runtime and commands in `INTEGRATION.md`; this uv environment reproduces model
training/inference and per-cohort metrics, not a changed aggregate estimator.

## Verification

```bash
uv run --locked pytest -q test_validation.py test_benchmark.py test_audit_benchmark.py
uv run --locked python check_benchmark_portability.py \
  --root "$R" --output "$R/evaluation/tabpfn-reproduction/reload-checks"
```

`UV_VERIFICATION.md` records fresh-environment results. Checks cover protocol and
feature guards, serialization checksums, and representative frozen/fine-tuned
reloads. The published 60-fit matrix is preserved; no model is refitted during
these checks. Numerical equivalence on this host does not guarantee bitwise
identity across GPU architectures.

## New, already featurized inputs

The same CLI accepts `--input features.csv` instead of `--partition`. Supply
unique `record_id`, unambiguous 19-nt guide `siRNA` and all 100 numeric columns
listed in the pinned feature manifest. Labels and assay metadata are unnecessary.
The feature order is read from the manifest, never inferred from CSV order;
nonfinite or missing features are rejected. The selected model continues to
condition on its saved training context without refitting on new rows.

```bash
uv run --locked python predict_saved.py \
  --root "$R" --model-dir /path/to/tabpfn35_finetuned \
  --input /path/to/new_records_features.csv --output /path/to/new_predictions
```

Feature production is a separate, existing step:

```bash
uv run --locked python ../regenerate_features.py \
  --input /path/to/new_records.csv --output /path/to/new_records_features.csv --workers 1
```

This feature generator requires `RNAfold`, `RNAcofold` and `RNAup` **2.4.18** on
PATH, matching the sealed benchmark manifest. These external executables are
not installed by uv. Follow the repository's strand/context contract and supply
`record_id`, antisense 19-nt `siRNA` and the central 19-nt target `mRNA`
(as in the corrected records). The feature generator uses `mRNA`, not the
57-nt `extended_mRNA` flanks.
The generator preserves metadata and rounds its 100 generated features to
three decimals. This packaging task tested prediction from the existing
canonical features stripped of labels/metadata; it did not rerun ViennaRNA or
validate unseen experimental inputs. The five fold-specific states serve
benchmark evaluation; this package does not silently pick a deployment fold.
