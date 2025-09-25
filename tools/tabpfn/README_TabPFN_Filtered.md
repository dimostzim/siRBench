# TabPFN siRBench Runner

This directory contains a single-purpose runner for the siRBench splits that ship with the repository. It fits a TabPFN regressor on the curated features and writes prediction CSVs for inspection.

## Expected Data

data/
├── siRBench_train_90.csv
├── siRBench_val_10.csv
└── siRBench_test.csv
```
Each file must keep the 107-column schema created by the existing feature pipeline.

## How to Run

```bash
cd tools/tabpfn
pip install uv
uv run tabpfn_sirbench.py
```
The first invocation bootstraps a lightweight environment, installs `tabpfn`, `pandas`, and `scipy`, and executes the script. If you prefer a manually managed environment, run `python tabpfn_sirbench.py` with the same working directory.

## Outputs

The script writes per-split predictions into `tools/tabpfn/results/`:
- `train_90_predictions.csv`
- `val_10_predictions.csv`
- `test_predictions.csv`

Each CSV contains two columns: `true` (ground truth efficacy) and `pred` (TabPFN predictions). The script prints the loss for the train and validation splits, plus Pearson correlation coefficients for the validation and test splits.

## Scope

The current implementation is tuned only for the bundled siRBench data. Running it on any other dataset is unsupported without modifying the code.
