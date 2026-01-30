# siRNABERT wrapper

## Prepare

```bash
python3 prepare.py --input-csv /path/to/train.csv --output-dir data --dataset-name train
```

## Train

```bash
python3 train.py --train-csv data/train.csv --val-csv data/val.csv --model-dir models/sirnabert
```

Training requires a validation set; run `test.py` separately on the held-out test set.

## Test

```bash
python3 test.py --test-csv data/test.csv --model-path models/sirnabert/model.pt --output-csv preds.csv
```

The Docker image includes the recommended DNABERT 6-mer weights at `/opt/dnabert/6mer`.
Use `--bert-dir` only if you want to override that path.
