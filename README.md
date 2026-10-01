# siRBench

A harmonized and reproducible benchmark for siRNA efficacy prediction on public
in-vitro measurements. This repository contains the code used for the revised
paper, with portable reproduction commands and pinned model environments.

**Version 2 code, data and prediction archive**
[10.5281/zenodo.23001225](https://doi.org/10.5281/zenodo.23001225).
The archive is the immutable experimental snapshot. Repository packaging changes
are documented in [REPRODUCING.md](REPRODUCING.md).

## What is included

- Dataset audit, the 702 Takayuki strand corrections, 100-feature generation,
  target/guide grouping and label-blind split construction. The corrected dataset
  has 4,098 records, including the full 1,047-record HeLa transfer cohort.
- Audited Docker wrappers for OligoFormer, GNN4siRNA, siRNADiscovery, AttSiOff,
  BERT-siRNA and ENsiRNA, with upstream commits and runtime package versions.
- Seven simple reference baselines and common evaluation code, including paired
  group-bootstrap uncertainty, source/cell-line summaries and sensitivity analyses.
- All five selected Agentomics reference pipelines, training/inference source,
  fitted artifacts and independent `uv.lock` files.
- Frozen and fine-tuned TabPFN-3.5 code with its own locked `uv` environment.
  Foundation and fitted TabPFN weights require separate authorized access.

The primary protocol has five grouped folds and three training seeds. Five
unselected random partitions are complementary. The Agentomics results use one
selected pipeline per grouped fold, not three-seed refits or a five-model
prediction ensemble. No source/cell-line covariates enter these revised models.

## Start here

From the repository root, with Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run --locked python scripts/fetch_archive.py
uv run --locked python scripts/reproduce_dataset.py \
  --package artifacts/siRBench-v2-2026-09-27 \
  --output outputs/dataset
```

This verifies the ZIP and extracted files, rebuilds the corrected records and
all partitions, and checks them against the archived hashes. The default reuses
the archived numeric features. To recompute all 100 features, install ViennaRNA
2.4.18 executables and use `--features regenerate --workers 8` with a new output
directory. The full feature rebuild has been checked against the archive.

On Linux x86-64, reproduce the numerical tables from saved predictions:

```bash
uv run --frozen --project reproduce python reproduce/replay_results.py \
  --package artifacts/siRBench-v2-2026-09-27 \
  --output outputs/result-replay
```

This includes TabPFN, all five Agentomics pipelines, 2,000 bootstrap draws and
all 12 paired sensitivity comparisons. It does not retrain models.

## Model execution

| Models | Environment | Instructions |
|---|---|---|
| Six published siRNA predictors | Separate Linux/NVIDIA Docker images | [Competitor reproduction](REPRODUCING.md#six-published-predictors) |
| Five selected Agentomics pipelines | One locked uv project per fold | [Training and saved-model inference](benchmark/revision/reference_models/README.md) |
| Frozen and fine-tuned TabPFN-3.5 | Locked uv project, Linux/NVIDIA | [Training and saved-state inference](benchmark/revision/tabpfn/UV_REPRODUCE.md) |
| Dataset, baselines and common analysis | Root/reproduce uv projects, CPU | [Full reproduction guide](REPRODUCING.md) |

Obtain the Agentomics checkpoint files after cloning with `git lfs install` followed by `git lfs pull`.
GitHub's source ZIP may contain LFS pointers; the Zenodo ZIP contains the actual
five selected models. Reproduction instructions distinguish saved-model
inference, refitting selected pipelines and repeating an Agentomics search.

The old `model/` predictor and `data/` splits are retained as historical inputs.
They are not the revised reference models or primary evaluation partitions.
Use the commands above and the version 2 archive for the revised paper.

## Scope and licensing

These legacy in-vitro data do not benchmark chemically modified therapeutics,
delivery systems or clinical formulations. Rankings depend on evaluation axis,
and uncertainty is conditional on fitted models and partitions.

First-party code is MIT licensed. Source datasets, upstream software and model
assets retain their own terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and [licenses/](licenses/). OligoFormer source/checkpoints, Rosetta binaries and
TabPFN weights are not redistributed here. The six competitors' trained
checkpoints are inventoried in `provenance/external-model-artifacts.csv` and are
not included in the public ZIP; all 228 runs' prediction evidence is archived.
