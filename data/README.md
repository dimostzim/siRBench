# Data

Run from the repository root.

## Setup and download

```bash
uv sync --locked --project data
uv run --locked --project data python data/scripts/download.py
```

The archive is downloaded to `data/archive/`. Use `--zip /path/to/archive.zip`
if it is already downloaded.

## Prepare the dataset

```bash
uv run --locked --project data python data/scripts/prepare.py \
  --package data/archive/siRBench-v2-2026-09-27 \
  --output data/processed
```

Prepares 4,098 records, five grouped splits, five random splits and the full
1,047-record HeLa set using the archived data and target mappings.
Numeric features are reused from the archive.
Add `--features regenerate --workers 8` to recompute them with `RNAfold`,
`RNAcofold` and `RNAup` 2.4.18 on PATH. Use a new output directory for each run.
Reconstruction starts from the harmonized archive. The complete curation script
from the raw publications is unavailable.

Outputs:

- `data/processed/datasets/corrected-v1/records_features.csv`
- `data/processed/evaluation/protocol-v1/{grouped,random}/fold_N/{train,val,test}.csv`
- `data/processed/evaluation/protocol-v1/hela_full.csv`
- `data/processed/reference-training/fold_N/{train,validation,test}/` for Agentomics.

See [benchmark/](../benchmark/README.md) to reproduce the result tables.
