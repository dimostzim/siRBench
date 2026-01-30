# siRNADiscovery wrapper

Requires preprocessing matrices and RNA-AGO2 features.

## Prepare

```bash
python3 prepare.py --input-csv /path/to/train.csv --output-dir data --dataset-name train \
  --preprocess-dir /path/to/siRNA_split_preprocess --rna-ago2-dir /path/to/RNA_AGO2
```

## Train

```bash
python3 train.py --train-csv data/train.csv --val-csv data/val.csv \
  --preprocess-dir data/siRNA_split_preprocess --rna-ago2-dir data/RNA_AGO2 \
  --model-dir models/sirnadiscovery
```

## Test

```bash
python3 test.py --test-csv data/test.csv --preprocess-dir data/siRNA_split_preprocess --rna-ago2-dir data/RNA_AGO2 \
  --model-path models/sirnadiscovery/model.keras --output-csv preds.csv
```
