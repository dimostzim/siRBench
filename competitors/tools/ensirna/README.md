# ENsiRNA wrapper

Requires PDB inputs and positions for each sample.
If `pdb_data_path` is missing, `prepare.py` will attempt to generate PDBs using Rosetta.
Set `ROSETTA_DIR` (or pass `--rosetta-dir`) to a Rosetta install that includes `rna_denovo`
and the Rosetta database. By default, we store Rosetta under `tools/ensirna/rosetta` and mount
it into the container at runtime.
To download and extract Rosetta into `tools/ensirna/rosetta`:

```bash
./fetch_rosetta.sh
```

The Rosetta database is required; the `rosettacommons/rosetta` image and the Graylab
binary-only archives do not include it.
`fetch_rosetta.sh` downloads the Rosetta 3.15 Ubuntu bundle from RosettaCommons and extracts
only the binaries and database needed by ENsiRNA.
Then build with `../../setup.sh --tool ensirna`. The container expects Rosetta to be
available at `ROSETTA_DIR` (default `tools/ensirna/rosetta`) and uses the host mount at runtime.
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
