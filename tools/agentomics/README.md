# Agentomics siRBench Runner

This folder bundles the pretrained Agentomics model and helpers to score the siRBench splits that ship with the repository.

## Expected Data

All required CSVs already live in `/home/dtzim01/igem_2025/data/` (relative to this directory: `../data/`):
```
data/
├── siRBench_train_90.csv
├── siRBench_val_10.csv
└── siRBench_test.csv
```

## How to Run

```bash
cd tools/agentomics
pip install uv
uv run agentomics_sirbench.py
```

The first run installs `lightgbm`, `pandas`, `numpy`, `scikit-learn`, `joblib`, and `scipy` into an isolated environment before executing the script. Prefer to manage the environment yourself? Activate it and replace `uv run` with `python` in the command above.

## Outputs

Predictions are written to `tools/agentomics/results/`:
- `train_90_predictions.csv`
- `val_10_predictions.csv`
- `test_predictions.csv`

Each CSV contains two columns: `true` (ground-truth efficacy when available) and `pred` (Agentomics predictions). The script prints the training and validation losses along with Pearson correlation coefficients for the validation and test splits.

## Scope

The current workflow targets the curated siRBench feature tables bundled with the repo. Running it on other datasets is unsupported without code changes.
