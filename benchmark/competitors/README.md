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

Use the wrapper to run prepare/train/test in one go (benchmark defaults: epochs=100, early-stop=20, metric=R2):

```bash
./run_tool.sh --tool oligoformer gnn4sirna sirnadiscovery \
  --train /path/to/train.csv \
  --val /path/to/val.csv \
  --test /path/to/test.csv \
  --seed 42 --deterministic
```

To evaluate an additional unseen set, pass `--leftout /path/to/leftout.csv`.
To run upstream defaults, add `--original` (results go to `benchmark/competitors/original_results/`).

## Prepare (standalone)

```bash
python3 scripts/prepare.py --tool oligoformer \
  --input-csv /path/to/train.csv \
  --output-dir data \
  --dataset-name train
```

Each tool has its own `prepare.py` with extra flags; see `tools/<tool>/README.md`.

## Train (standalone)

```bash
python3 scripts/train.py --tool oligoformer \
  --train-csv data/train.csv \
  --val-csv data/val.csv \
  --data-dir data \
  --model-dir models/oligoformer
```

All tools expect an explicit validation set for training. Run `test.py` separately on a held-out test set.

## Test (standalone)

```bash
python3 scripts/test.py --tool oligoformer \
  --test-csv data/test.csv \
  --data-dir data \
  --model-path models/oligoformer/model.pt \
  --output-csv ../updated_validation_results/oligoformer/preds.csv
```

`scripts/test.py` prints common regression metrics (MAE, MSE, RMSE, R2, Pearson, Spearman).
Use `--metrics-json /path/to/metrics.json` to save the metrics to disk.

## After testing

Outputs are written under `models/` and `updated_validation_results/` by default:

- `models/<tool>/` contains trained model weights (e.g., `model.pt` or `model.keras`).
- `updated_validation_results/<tool>/preds.csv` contains predictions for the test set.
- `updated_validation_results/<tool>/metrics.json` contains the regression metrics for that run.

If you run with `--original`, outputs go to `original_results/`.

## Plot metrics

Generate a 3x3 panel PNG (one tool per panel) from the saved metrics:

```bash
python3 scripts/plot_metrics.py
```

The output is written to `updated_validation_results/metrics_panels.png` by default; pass `--results-dir ../original_results` to plot original runs.

## siRNADiscovery: RPISeq / AGO2 inputs

siRNADiscovery requires external AGO2 features from the RPISeq web tool. You must provide:

```
benchmark/competitors/data/sirnadiscovery/RNA_AGO2/
  ├─ siRNA_AGO2.csv
  └─ mRNA_AGO2.csv
```

Expected format for each file:
- CSV with an ID column (index) and a single numeric column named `RF_Classifier_prob`.
- The IDs must match the hashed `siRNA` / `mRNA` IDs produced by the siRNADiscovery `prepare.py`.

To generate batches for RPISeq submission from a prepared CSV (with columns `siRNA`, `mRNA`, `siRNA_seq`, `mRNA_seq`):

```bash
python3 tools/sirnadiscovery/scripts/rpiseq_export.py \
  --input-csv /path/to/prepared.csv \
  --out-dir /tmp/rpiseq_batches --type sirna

python3 tools/sirnadiscovery/scripts/rpiseq_export.py \
  --input-csv /path/to/prepared.csv \
  --out-dir /tmp/rpiseq_batches --type mrna
```

After running RPISeq, convert the output tables:

```bash
python3 tools/sirnadiscovery/scripts/rpiseq_convert.py \
  --input /path/to/rpiseq_output.csv \
  --output benchmark/competitors/data/sirnadiscovery/RNA_AGO2/siRNA_AGO2.csv

python3 tools/sirnadiscovery/scripts/rpiseq_convert.py \
  --input /path/to/rpiseq_output.csv \
  --output benchmark/competitors/data/sirnadiscovery/RNA_AGO2/mRNA_AGO2.csv
```

The converter will infer the RF classifier column (or use `--rf-col` if needed) and strip any leading `>` from IDs.
