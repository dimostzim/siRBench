# siRNA Data Processing Pipeline

Process siRNA datasets with feature calculation, harmonization, and splitting utilities.

## Usage

### Feature Calculation
Builds the enhanced feature set combining composition, constraint‑based energies, and RNAup.

```bash
# Default: reads data/siRBench_base.csv and writes data/siRBench_enhanced.csv
python data/scripts/make_all_features.py

# Explicit paths
python data/scripts/make_all_features.py data/siRBench_base.csv -o data/siRBench_full.csv

# Performance/ablation toggles
python data/scripts/make_all_features.py \
  [--no-advanced-energies] \
  [--no-rnaup] \
  [--no-cofold] \
  [--with-std-seqs]
```

- Input must contain exact `siRNA` and `mRNA` columns (19 nt expected; `T`→`U`, truncated to 19).
- Outputs include:
  - Composition/NN: `ends`, `DH_all`, base fractions (`U_all`, `G_all`) and dinucleotide fractions (`UU_all`, `GG_all`, `GC_all`, `CC_all`, `UA_all`).
  - Constraint energies (RNAfold/RNAcofold): `single_energy_total`, `single_energy_pos1..19`, `duplex_energy_total`, `duplex_energy_sirna_pos1..19`, `duplex_energy_target_pos1..19`.
  - RNAup: `RNAup_open_dG` (opening siRNA+target), `RNAup_interaction_dG` (hybridization).
  - All numeric columns are rounded to 2 decimals. Optional standardized sequences with `--with-std-seqs`.

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

- `data/scripts/features_calculator.py`: Core composition ("oligoformer") feature calculations.
- `data/scripts/make_all_features.py`: Build enhanced feature tables.
- `data/scripts/stratified_split.py`: Stratified 90/10 train/val split for siRBench.
