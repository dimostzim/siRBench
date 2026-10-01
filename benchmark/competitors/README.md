# siRBench competitors

Docker wrappers for OligoFormer, GNN4siRNA, siRNADiscovery, AttSiOff, BERT-siRNA
and ENsiRNA. Use Linux x86-64 with Docker, NVIDIA Container Toolkit and a GPU.
The setup scripts pin upstream commits; Dockerfiles pin the observed package
versions. Existing checkouts at different commits are rejected.

## Setup

From the repository root:

```bash
tool=gnn4sirna
IMAGE_TAG="$tool:revision" bash "benchmark/competitors/tools/$tool/setup.sh" --docker
export SIRBENCH_IMAGE_TAG=revision
```

Tool names are `oligoformer`, `gnn4sirna`, `sirnadiscovery`, `attsioff`,
`sirnabert` and `ensirna`. RNA-FM is downloaded by the relevant setup scripts.
ENsiRNA also obtains the pinned Rosetta release 371 runtime and uses ViennaRNA
2.6.4; authorized access and the upstream terms apply.

## Train and evaluate

Prepare the corrected data using [the data scripts](../../data/scripts/README.md).

```bash
export PROTOCOL="$PWD/data/processed/evaluation/protocol-v1"
SIRBENCH_IMAGE_TAG=revision uv run --locked --project data \
  bash benchmark/competitors/run_tool.sh \
  --tool gnn4sirna --seed 0 \
  --train "$PROTOCOL/grouped/fold_0/train.csv" \
  --val "$PROTOCOL/grouped/fold_0/val.csv" \
  --test "$PROTOCOL/grouped/fold_0/test.csv" \
  --leftout "$PROTOCOL/hela_full.csv" \
  --run-dir "$PWD/benchmark/competitors/runs/gnn4sirna_grouped0_seed0"
```

Use a new run directory under the repository for each tool, axis, fold and seed.
The primary comparison uses a 100-epoch cap and validation-R² stopping with
patience 20. Results go under the run directory's `results/<tool>/`; fitted
models go under `models/<tool>/`. The `prepare.py`, `train.py` and `test.py`
files provide separate preparation, training and saved-model inference commands.
Use `--help` for their arguments.

For siRNADiscovery, copy the frozen RPISeq features before running:

```bash
mkdir -p data/sirnadiscovery
cp -R data/archive/siRBench-v2-2026-09-27/workspace/datasets/corrected-v1/RNA_AGO2 \
  data/sirnadiscovery/
```

Graph wrappers use training-only neighborhoods and isolated-query inference.
The upstream OligoFormer and AttSiOff batch/order-dependent behaviors are
preserved in the primary comparison. Keep prediction order and batch sizes fixed.
The separate input-extent and original-schedule experiments remain in the Zenodo
snapshot; `--original` is not used for the main comparison.
