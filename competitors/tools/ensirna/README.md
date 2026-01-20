# ENsiRNA wrapper

Requires PDB inputs and positions for each sample.
If `pdb_data_path` is missing, `prepare.py` will attempt to generate PDBs using Rosetta.
Set `ROSETTA_DIR` (or pass `--rosetta-dir`) to a Rosetta install that includes `rna_denovo`.
The Docker image includes RNA-FM pretrained weights and ENsiRNA dependencies.

## Prepare

```bash
python3 prepare.py --input-csv /path/to/train.csv --output-jsonl data/train.jsonl
```

## Train

```bash
python3 train.py --train-set data/train.jsonl --valid-set data/val.jsonl --model-dir models/ensirna --gpus 0
```

## Test

```bash
python3 test.py --test-set data/test.jsonl --ckpt models/ensirna/*.ckpt --output-csv preds.csv
```
