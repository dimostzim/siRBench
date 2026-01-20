# OligoFormer wrapper

## Prepare

```bash
python3 prepare.py --input-csv /path/to/train.csv --output-dir data --dataset-name train
python3 prepare.py --input-csv /path/to/test.csv --output-dir data --dataset-name test
```

RNA-FM embeddings are generated automatically if missing. You can force regeneration with `--run-rnafm`.

```bash
python3 prepare.py --input-csv /path/to/train.csv --output-dir data --dataset-name train --run-rnafm --rnafm-root oligoformer_src/RNA-FM
```

The Docker image includes the RNA-FM pretrained weights; no runtime download is required.

## Train

```bash
python3 train.py --train-csv data/train.csv --val-csv data/val.csv --data-dir data --model-dir models/oligoformer
```

Training requires a validation set; run `test.py` separately on the held-out test set.

## Test

```bash
python3 test.py --test-csv data/test.csv --data-dir data --model-path models/oligoformer/model.pt --output-csv preds.csv
```
