# siRBench-model

Baseline model for siRBench (XGBoost + LightGBM with calibration).

## Setup (uv)

```bash
cd /home/dtzim01/siRBench/siRBench-model
uv venv
uv sync
```

## Train

Training expects an `efficacy` column in the input CSVs.

```bash
uv run python train.py \
  --train-data ../data/val_split/siRBench_train_split.csv \
  --validation-data ../data/val_split/siRBench_val_split.csv \
  --artifacts-dir training_artifacts
```

## Inference

```bash
uv run python inference.py \
  --input ../data/siRBench_test.csv \
  --output preds.csv \
  --artifacts-dir training_artifacts
```

## Data requirements

Required columns:
- `siRNA`
- `mRNA`
- `efficacy` (train only)

Optional columns:
- `id` (passed through to output)
- `source`, `cell_line` (categorical features)
- `extended_mRNA`
- any extra numeric feature columns (used automatically)

## Artifacts

Training writes:
- `feature_artifacts.json`
- `xgb_model.json`
- `lgbm_model.txt`
- `calibrator.joblib`

These are required for inference.
