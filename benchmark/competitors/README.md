# siRBench competitors

Unified wrappers to prepare/train/test competitor models. Commands run inside Docker by default.

GPU is required; Docker runs with `--gpus all`.
Images bake required pretrained weights (RNA-FM, DNABERT for siRNABERT) to avoid runtime downloads.
ENsiRNA `prepare.py` will generate PDBs if `pdb_data_path` is missing and Rosetta is available.

## Setup

Setup builds the Docker image and clones the upstream tool repository into each `tools/<tool>` directory.
Use `--tool <name>...` to limit setup to specific tools.

```bash
./setup.sh
```

## Run tools

Use the wrapper to run prepare/train/test in one go:

```bash
./run_tool.sh --tool oligoformer gnn4sirna sirnadiscovery \
  --train /path/to/train.csv \
  --val /path/to/val.csv \
  --test /path/to/test.csv
```

To evaluate an additional unseen set, pass `--leftout /path/to/leftout.csv`.

## Prepare

```bash
python3 prepare.py --tool oligoformer --input-csv /path/to/train.csv --output-dir data --dataset-name train
```

Each tool has its own `prepare.py` with extra flags; see `tools/<tool>/README.md`.

## Train

```bash
python3 scripts/train.py --tool oligoformer --train-csv data/train.csv --val-csv data/val.csv --data-dir data --model-dir models/oligoformer
```

All tools expect an explicit validation set for training. Run `test.py` separately on a held-out test set.

## Test

```bash
python3 scripts/test.py --tool oligoformer --test-csv data/test.csv --data-dir data --model-path models/oligoformer/model.pt --output-csv preds.csv
```

`scripts/test.py` prints common regression metrics (MAE, MSE, RMSE, R2, Pearson, Spearman).
Use `--metrics-json /path/to/metrics.json` to save the metrics to disk.

## After testing

Outputs are written under `models/` and `results/`:

- `models/<tool>/` contains trained model weights (e.g., `model.pt` or `model.keras`).
- `results/<tool>/preds.csv` contains predictions for the test set.
- `results/<tool>/metrics.json` contains the regression metrics for that run.

## Plot metrics

Generate a 3x3 panel PNG (one tool per panel) from the saved metrics:

```bash
python3 scripts/plot_metrics.py
```

The output is written to `results/metrics_panels.png`.

## Smoke test

Build images and verify each tool entrypoint runs:

```bash
./smoke_test.sh
```
