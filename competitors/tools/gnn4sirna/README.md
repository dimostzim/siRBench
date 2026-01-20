# GNN4siRNA wrapper

## Prepare

```bash
python3 prepare.py --input-csv /path/to/train.csv --output-dir data --dataset-name train
```

Optional preprocessing (requires ViennaRNA/RNAup):

```bash
python3 prepare.py --input-csv /path/to/train.csv --output-dir data --dataset-name train --run-preprocess
```

Preprocessing runs automatically if processed files are missing; use `--run-preprocess` to force regeneration.

## Train

```bash
python3 train.py --train-csv data/train.csv --val-csv data/val.csv --processed-dir data/processed/train --model-dir models/gnn4sirna
```

Training requires a validation set; `--test-csv` is not accepted.

## Test

```bash
python3 test.py --test-csv data/test.csv --processed-dir data/processed/train --model-path models/gnn4sirna/model.keras --output-csv preds.csv
```
