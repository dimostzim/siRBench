# AttSiOff wrapper

Requires additional feature columns: `s-Biopredsi`, `DSIR`, `i-score`.

## Prepare

```bash
python3 prepare.py --input-csv /path/to/train.csv --output-dir data
```

RNA-FM embeddings are generated automatically if missing. They are stored under `output-dir/data/RNAFM_*`
to match the upstream loader. You can force regeneration with `--run-rnafm`.

```bash
python3 prepare.py --input-csv /path/to/train.csv --output-dir data --run-rnafm --rnafm-root attsioff_src/RNA-FM
```

The Docker image includes the RNA-FM pretrained weights; no runtime download is required.

## Train

```bash
python3 train.py --train-csv data/train.csv --val-csv data/val.csv --model-dir models/attsioff
```

Training requires a validation set; run `test.py` separately on the held-out test set.

## Test

```bash
python3 test.py --test-csv data/test.csv --model-path models/attsioff/model.pt --output-csv preds.csv
```
