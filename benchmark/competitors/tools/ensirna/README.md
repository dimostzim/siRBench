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

The Linux installer retrieves the authors' Rosetta release-371 runtime
(`2024.09+release.06b3cf8`) from the immutable image
`tanwenchong/ensirna@sha256:1a9c8b80a2d5b5943770fb5e736264cb5234997181df98d7704bde673f09167f`.
It verifies the image manifest, application layer, operational file tree,
symlink targets, and both executables. It preserves the accompanying license and
writes provenance, hash, and actual-version sidecars beside the runtime directory.
The selected database, binaries and RNA utilities are byte-identical to those used
for revised canonical preparation; upstream source/test files are not needed.
Existing runtime directories must match the pin and are never overwritten.

A previously downloaded layer and its **raw** image manifest can be verified and
reused without another network transfer:

```bash
ROSETTA_LAYER_ARCHIVE=/path/to/app-layer.tar \
ROSETTA_IMAGE_MANIFEST=/path/to/image-manifest.raw.json \
ROSETTA_OUT_DIR=/path/to/rosetta-release-371 ./fetch_rosetta.sh
```

The old mutable RosettaCommons URL labeled 3.15 actually supplied release-408
during the audit; it is no longer a setup default or an automatic fallback.
Fresh source clones are also pinned to the audited ENsiRNA commit
`028824341635903f3c661f5d1cc737de106493d5`.
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

## Revision protocol and cache reuse

The wrapper now defaults to the released 1e-4/1e-5 learning-rate bounds, 100 epochs,
128-dimensional embedding, 256 hidden
units, two layers, nine neighbors, and shuffled training. `--original-params`
uses the released learning rates and 100 epochs, and preserves the original
stopping semantics for a labeled sensitivity run. The main benchmark uses global
validation R² and patience against the best value seen so far.

The image pins both ViennaRNA CLI and Python bindings to 2.6.4. Preparation uses
RNAplex-derived secondary structure, a fixed Rosetta preprocessing seed, and
fails instead of silently omitting rows. `record_id` is preserved automatically.
`--pdb-dir` can point to a versioned shared directory; `--pdb-workers` defaults to 4.
Internal unknown bases remain in their original positions.

`subset_features.py --freeze` records dependencies of a canonical deterministic
feature cache. Subsetting verifies the frozen files and sequences, then injects
current split labels by record ID. Cache reuse never fits across evaluation rows.
`run_matrix.py` executes the frozen five-fold grouped and random protocols with
three seeds, isolated model/result directories, and full HeLa evaluation.

See `benchmark/revision/ENSIRNA_AUDIT.md` for the official-source comparison,
representation changes, regression checks, and remaining validation.
