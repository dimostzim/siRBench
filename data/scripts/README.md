# Data Scripts

Quick reference for helper scripts in `data/scripts/`.

## make_all_features.py
- Build the full feature table (thermo + RNAup + sequence features).
- Input: CSV with `siRNA` and `mRNA` columns (default `data/siRBench_base.csv`).
- Output: CSV with added feature columns (default `data/siRBench_with_features.csv`).
- Usage:
  ```bash
  python3 make_all_features.py
  python3 make_all_features.py ../siRBench_full_base.csv -o ../siRBench_with_features.csv
  ```

## features_calculator.py
- Utility functions for thermodynamic features (no CLI).
- Input/Output: Imported by `make_all_features.py`.

## ecdf_visualization.py
- Plot CDFs for train/test/leftout efficiency distributions.
- Input: `data/siRBench_train.csv`, `data/siRBench_test.csv`, `data/leftout/siRBench_leftout.csv`.
- Output: `data/plots/ecdf_comparison.png`.

## KS_&_p-value_calculation.py
- KS test between train and leftout efficiency distributions.
- Input: `data/siRBench_train.csv`, `data/leftout/siRBench_leftout.csv`.
- Output: Prints KS statistic and p-value to stdout.

## split_train_val.py
- Split a CSV into train/val with stratification (CLI).
- Input: `--input-csv <path>` (expects `cell_line` and `efficiency` if using default stratify cols).
- Output: `--train-out <path>`, `--val-out <path>`.
- Usage:
  ```bash
  python3 split_train_val.py --input-csv ../siRBench_train.csv \
    --train-out ../val_split/siRBench_train_split.csv \
    --val-out ../val_split/siRBench_val_split.csv
  ```

## training_testing_split.py
- Make a train/test split that minimizes distribution shift; prints metrics and shows a CDF plot.
- Input: `data/siRBench_train.csv`.
- Output: Overwrites `data/siRBench_train.csv` and `data/siRBench_test.csv`; prints metrics.
- Note: Run with caution since it overwrites existing files.

## find_dropped_leftout.py
- Identify leftout rows dropped during filtering.
- Input: `data/leftout/siRBench_hela.csv`, `data/leftout/siRBench_leftout.csv`.
- Output: `data/leftout/siRBench_leftout_dropped.csv`.
