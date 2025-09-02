# siRNA Data Processing Pipeline

Process siRNA datasets with feature calculation, harmonization, and splitting utilities.

## Usage

### Feature Calculation
```bash
python data/scripts/add_features.py input.csv --output results.csv
```

- Input must contain `siRNA` and `mRNA` (or `target`) columns. Case-insensitive; `T` is converted to `U`.
- Adds 24 thermodynamic features and `duplex_folding_dG` to the CSV.

### Train/Validation Split (siRBench)
Create a 90/10 split from `data/siRBench_train.csv` while preserving distributions across cell line, binary label, and binned efficacy.

```bash
python data/scripts/stratified_split.py \
  --input data/siRBench_train.csv \
  --train_out data/siRBench_train_90.csv \
  --val_out data/siRBench_val_10.csv \
  --test_size 0.10 \
  --bins 5 \
  --seed 42
```

- Requires columns: `cell_line`, `binary`, `efficacy`.
- Stratifies by `(cell_line, binary, eff_bin)` where `eff_bin` is a per–cell-line rank-quantile bin.
- Very small strata (size 1) are kept entirely in train to avoid empty categories.

## Scripts

- `data/scripts/features_calculator.py`: Core feature calculations.
- `data/scripts/add_features.py`: Process CSV files with siRNA/target data.
- `data/scripts/stratified_split.py`: Stratified 90/10 train/val split for siRBench.
