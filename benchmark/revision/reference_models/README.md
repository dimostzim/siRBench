# siRBench reference models

Five independently selected, sequence-based pipelines from the corrected siRBench grouped-fold searches. Each directory contains its own training source, inference source, fitted preprocessing, checkpoints, pinned Python dependencies, and `uv.lock`. No model requires files from another fold or search iteration. Source and cell-line covariates are not inputs.

| Grouped fold | Selected iteration (zero-based) | Completed iterations | Selected pipeline | Predictors |
|---|---:|---:|---|---:|
| 0 | 16 | 19 | Polynomial kernel ridge + sequence CNN with joint linear skip + CatBoost | 3 |
| 1 | 17 | 20 | RBF SVR with sequence and ViennaRNA features | 1 |
| 2 | 15 | 18 | CNN with joint positional branch | 1 |
| 3 | 17 | 18 | Sequence CNN + thermodynamic CNN + CatBoost, with training-only OOF affine calibration | 3 |
| 4 | 13 | 20 | CatBoost + ExtraTrees | 2 |

Each search had a budget of 20 iterations, including three exploration iterations, and selected its best completed pipeline by that fold's validation R². Provider usage limits ended three searches before the full budget: 95 iterations completed in total. The accepted checkpoints are from the searches with base training seed 0. These results use one selected pipeline per fold; they are not three-seed refits. The five pipelines were frozen before held-out evaluation. Their differing architectures must not be ranked against one another using different test folds.

The grouped result concatenates one prediction per non-HeLa record from the pipeline whose test fold contains that record (3,051 records). The full HeLa result averages the five individual pipelines' metrics on the same 1,047 rows; it does not average their predictions into a five-model ensemble. Grouped out-of-fold Pearson r / R² are 0.62345053 / 0.38829343; mean full-HeLa r / R² are 0.59795371 / 0.26220271. These summaries describe this fold-specific search procedure, rather than a single globally selected architecture. No unselected-random-split run was performed for these reference models.

## Predict with an accepted checkpoint

Use Linux x86-64, Python 3.12, `uv`, and Git LFS. After cloning, obtain the repository's LFS objects with `git lfs install` followed by `git lfs pull`. A compatible CUDA GPU is used automatically by neural models when available; CPU inference is supported. There is no LLM call or network access in model inference after dependencies and checkpoints have been installed.

From this directory:

```bash
uv sync --frozen --project fold_0
uv run --frozen --project fold_0 python fold_0/inference.py \
  --input /absolute/path/to/input_directory \
  --output /absolute/path/to/predictions.csv
```

Replace `fold_0` with the desired fold. The input directory must contain `data.csv`, with exactly the benchmark input contract:

- `id`: stable row identifier, used only to join/output predictions.
- `siRNA`: the harmonized 19-nucleotide guide sequence.
- `extended_mRNA`: the harmonized 57-nucleotide target context, including the central 19-nucleotide target site.

Use the corrected benchmark's orientation and nucleotide conventions. Do not supply labels or assay metadata to inference. Output is a CSV with `id,prediction`; predictions are continuous efficacy values and must not be silently clipped when reproducing reported metrics. The default artifact directory is the selected fold's bundled `training_artifacts`; `--artifacts-dir` provides an explicit override. Create the output parent directory before invoking fold 4.

## Training source

The original selected iteration's executable training source is included for transparency and reuse. For example:

```bash
uv run --frozen --project fold_0 python fold_0/train.py \
  --train-data /absolute/path/to/grouped_fold_0/train \
  --validation-data /absolute/path/to/grouped_fold_0/validation \
  --artifacts-dir /absolute/path/to/new_run/training_artifacts
```

Each split directory contains `input/data.csv` and `labels.csv` (`id,label`). Always use a new output directory: training writes diagnostics alongside the requested artifacts. These scripts retain the selected iteration's internal candidate evaluations and validation checkpoint selection; they do not implement a new 20-iteration Agentomics search. Re-running them is distinct from reproducing the saved checkpoint's predictions and is not asserted to be bitwise deterministic. No training was rerun during packaging. Neural fitting is capped at 100 epochs with patience 20 and validation R² checkpoint selection. See the fold-specific source/configuration for the full candidate settings and tree/kernel training procedures.

## Provenance and verification

`manifest.json` records source/artifact SHA256 hashes, selected iterations, seeds, and completed iteration counts. `original_environment.yml` preserves each search environment as provenance. The smaller `pyproject.toml` and `uv.lock` define the clean, tested runtime; Python scientific package versions directly used by each model are pinned. No archived Conda environment, search history, credentials, or sibling models are required.

Only files required by active training/inference and their configuration are packaged. Unused checkpoint files left in Agentomics snapshot overlays are excluded. Fold 1's inference default was changed from an absolute Agentomics workspace path to its bundled local artifact directory; the learned model is unchanged. All other copied active source and checkpoint bytes are preserved. The package contains approximately 139 MB of model artifacts, managed by the package-local Git LFS patterns.

`verification.json` records clean-`uv` inference parity against the already frozen validation, grouped-test, and full-HeLa predictions for all five models. Verification checks artifact/source hashes, training entry-point imports, exact prediction row order, finite predictions, and maximum absolute prediction difference ≤ 1e-6. It exercises saved-model inference, not retraining.

For the canonical benchmark evaluation archive, the verification command is:

```bash
python verify_package.py \
  --evaluation-root /path/to/sirbench-selected-evaluation-20260923 \
  --output-dir /path/to/new_verification_outputs
```

The archive layout is `fold_N/inputs/{validation,test,test_hela_full}/data.csv` and `fold_N/predictions/{validation,test,test_hela_full}.csv`. The public archive contains this evaluation layout at `reference-evaluation/`. The immutable source/model snapshot is [Zenodo version 2](https://doi.org/10.5281/zenodo.23001225).
