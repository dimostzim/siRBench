# siRBench competitors

Unified wrappers to prepare/train/test competitor models. Commands run inside Docker by default.

GPU is required; Docker runs with `--gpus all`.
Images bake required pretrained weights (RNA-FM, DNABERT for siRNABERT) to avoid runtime downloads.
ENsiRNA `prepare.py` will generate PDBs if `pdb_data_path` is missing and Rosetta is available.

## Setup

Setup builds the Docker image and clones the upstream tool repository into each `tools/<tool>` directory.
Use `--tool <name>...` to limit setup to specific tools.

```bash
./benchmark/competitors/setup.sh
```

## Run tools

Use the wrapper to run prepare/train/test in one go:

```bash
./benchmark/competitors/run_tool.sh --tool oligoformer gnn4sirna sirnadiscovery \
  --train /path/to/train.csv \
  --val /path/to/val.csv \
  --test /path/to/test.csv
```

To evaluate an additional unseen set, pass `--leftout /path/to/leftout.csv`.

## Prepare

```bash
python3 benchmark/competitors/scripts/prepare.py --tool oligoformer \
  --input-csv /path/to/train.csv \
  --output-dir benchmark/competitors/data \
  --dataset-name train
```

Each tool has its own `prepare.py` with extra flags; see `tools/<tool>/README.md`.

## Train

```bash
python3 benchmark/competitors/scripts/train.py --tool oligoformer \
  --train-csv benchmark/competitors/data/train.csv \
  --val-csv benchmark/competitors/data/val.csv \
  --data-dir benchmark/competitors/data \
  --model-dir benchmark/competitors/models/oligoformer
```

All tools expect an explicit validation set for training. Run `test.py` separately on a held-out test set.

## Test

```bash
python3 benchmark/competitors/scripts/test.py --tool oligoformer \
  --test-csv benchmark/competitors/data/test.csv \
  --data-dir benchmark/competitors/data \
  --model-path benchmark/competitors/models/oligoformer/model.pt \
  --output-csv benchmark/results/oligoformer/preds.csv
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
python3 benchmark/competitors/scripts/plot_metrics.py
```

The output is written to `results/metrics_panels.png`.
