siRBench train/validation split

- Goal: Create a 90/10 train/validation split from `data/siRBench_train.csv` that preserves cell line balance, binary label balance, and approximates the efficacy distribution in each subset.

Inputs and outputs
- Input: `data/siRBench_train.csv`
- Outputs: `data/siRBench_train_90.csv` (90%), `data/siRBench_val_10.csv` (10%)

Method
- Efficacy was binned per cell line using rank-based quantiles into 5 bins (`--bins 5`).
- A composite stratification key `(cell_line, binary, eff_bin)` was used to split, maintaining distribution across cell lines, labels, and efficacy ranges.
- Random seed `42` ensures reproducibility. Very small strata (size 1) are left entirely in train to avoid empty categories.

Reproduce
```
python3 data/scripts/stratified_split.py \
  --input data/siRBench_train.csv \
  --train_out data/siRBench_train_90.csv \
  --val_out data/siRBench_val_10.csv \
  --test_size 0.10 \
  --bins 5 \
  --seed 42
```

Summary (current split)
- Train: 3068 rows (90.0%)
  - h1299(0)=1069, h1299(1)=1057, hela(0)=625, hela(1)=317
- Val: 340 rows (10.0%)
  - h1299(0)=118, h1299(1)=117, hela(0)=70, hela(1)=35

Notes
- Adjust `--bins` to coarser/finer efficacy control; change `--seed` for a different random split.
