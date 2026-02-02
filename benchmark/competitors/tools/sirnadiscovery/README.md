# siRNADiscovery wrapper

Requires preprocessing matrices (RNAfold/RNAcofold) and RNA-AGO2 features (RPISeq).

## Prepare

```bash
python3 prepare.py --input-csv /path/to/train.csv --output-dir data --dataset-name train \
  --preprocess-dir /path/to/siRNA_split_preprocess --rna-ago2-dir /path/to/RNA_AGO2
```

To generate preprocess matrices locally (requires ViennaRNA), use:

```bash
python3 prepare.py --input-csv /path/to/train.csv --output-dir data --dataset-name train \
  --run-preprocess --sirna-len 19 --mrna-len 57
```

For strict evaluation, generate preprocess matrices separately for train/val and test/leftout.

## RPISeq (AGO2) export + convert

RPISeq requires a protein sequence and RNA FASTA input (max 100 RNAs per batch).
Use the exporter to create FASTA batches from a prepared CSV:

```bash
python3 scripts/rpiseq_export.py --input-csv data/train.csv --out-dir rpiseq_batches --type sirna
python3 scripts/rpiseq_export.py --input-csv data/train.csv --out-dir rpiseq_batches --type mrna
```

After running the RPISeq batch tool, convert its output table to the expected format:

```bash
python3 scripts/rpiseq_convert.py --input rpi_output.csv --output data/RNA_AGO2/siRNA_AGO2.csv
python3 scripts/rpiseq_convert.py --input rpi_output.csv --output data/RNA_AGO2/mRNA_AGO2.csv
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
