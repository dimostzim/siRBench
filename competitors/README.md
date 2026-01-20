# siRBench competitors

Unified wrappers to prepare/train/test competitor models. Commands run inside Docker by default.

GPU selection: use `--gpus all` (default) or `--gpus 0` to select a device.
Images bake required pretrained weights (RNA-FM, DNABERT for siRNABERT) to avoid runtime downloads.
ENsiRNA `prepare.py` will generate PDBs if `pdb_data_path` is missing and Rosetta is available.

## Setup

Setup builds the Docker image and clones the upstream tool repository into each `tools/<tool>` directory.

```bash
./setup.sh --tool oligoformer
./setup.sh --tool gnn4sirna
./setup.sh --tool sirnadiscovery
./setup.sh --tool attsioff
./setup.sh --tool sirnabert
./setup.sh --tool ensirna
```

## Prepare

```bash
python3 prepare.py --tool oligoformer --input-csv /path/to/train.csv --output-dir data --dataset-name train
```

Each tool has its own `prepare.py` with extra flags; see `tools/<tool>/README.md`.

## Train

```bash
python3 train.py --tool oligoformer --train-csv data/train.csv --val-csv data/val.csv --data-dir data --model-dir models/oligoformer
```

All tools expect an explicit validation set for training. Run `test.py` separately on a held-out test set.

## Test

```bash
python3 test.py --tool oligoformer --test-csv data/test.csv --data-dir data --model-path models/oligoformer/model.pt --output-csv preds.csv
```

`test.py` prints common regression metrics (MAE, MSE, RMSE, R2, Pearson, Spearman).
Use `--metrics-json /path/to/metrics.json` to save the metrics to disk.

## Smoke test

Build images and verify each tool entrypoint runs:

```bash
./smoke_test.sh
```
