# AGENT.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

siRBench is a benchmark framework for evaluating computational methods for predicting siRNA (small interfering RNA) knockdown efficacy. It provides:
- A curated dataset of 4,098 siRNA-target pairs across 7 human cell lines
- A standardized 1,000-sample benchmark split (800 train / 100 val / 100 test)
- Docker-based wrappers for 6 different prediction methods with unified interfaces
- Thermodynamic feature calculations using ViennaRNA utilities

## Key Commands

### Setup and Build
```bash
# Build Docker image for a specific tool
./competitors/setup.sh --tool oligoformer

# Available tools: oligoformer, gnn4sirna, sirnadiscovery, attsioff, sirnabert, ensirna

# Verify all tools are working
./competitors/smoke_test.sh
```

### Running the Full Pipeline (OligoFormer example)
```bash
# Run complete prepare → train → test pipeline
./competitors/run_oligoformer.sh

# Or with custom paths:
DATA_ROOT=/path/to/data OUT_DATA=/path/to/output ./competitors/run_oligoformer.sh
```

### Individual Steps
```bash
# Prepare data (converts to tool-specific format)
python3 competitors/prepare.py --tool oligoformer \
  --input-csv data/siRBench_full_base_1000_train.csv \
  --output-dir competitors/data/oligoformer \
  --dataset-name train

# Train model
python3 competitors/train.py --tool oligoformer \
  --train-csv competitors/data/oligoformer/train.csv \
  --val-csv competitors/data/oligoformer/val.csv \
  --data-dir competitors/data/oligoformer \
  --model-dir competitors/models/oligoformer

# Test and evaluate
python3 competitors/test.py --tool oligoformer \
  --test-csv competitors/data/oligoformer/test.csv \
  --data-dir competitors/data/oligoformer \
  --model-path competitors/models/oligoformer/model.pt \
  --output-csv competitors/preds.csv \
  --metrics-json competitors/metrics.json
```

### Feature Generation (requires ViennaRNA)
```bash
cd data/scripts
python3 make_all_features.py ../siRBench_full_base_1000.csv -o ../siRBench_with_features.csv
```

## Architecture

### Directory Structure
```
siRBench/
├── data/                           # Datasets and feature utilities
│   ├── siRBench_full_base_1000_*.csv  # Train/val/test splits
│   └── scripts/                    # ViennaRNA feature calculators
└── competitors/                    # Unified ML pipeline
    ├── runner.py                   # Docker orchestration (path translation, GPU)
    ├── prepare.py                  # Wrapper → tools/<tool>/prepare.py
    ├── train.py                    # Wrapper → tools/<tool>/train.py
    ├── test.py                     # Wrapper → tools/<tool>/test.py
    ├── metrics.py                  # Evaluation (MAE, MSE, RMSE, R², Pearson, Spearman)
    └── tools/<tool>/               # Tool-specific implementations
```

### Docker Execution Model

All commands execute inside Docker containers via `runner.py`:
- Host paths are automatically translated to container paths (`/work/...`)
- GPU support enabled by default (`--gpus all`)
- Repository mounted at `/work` in container
- Tool-specific images include pre-downloaded weights (RNA-FM, DNABERT)

### Supported Tools

| Tool | Method | Pre-downloaded Weights |
|------|--------|----------------------|
| oligoformer | RNA-FM embeddings + thermodynamics | RNA-FM |
| sirnabert | DNABERT sequence model | DNABERT 6-mer |
| attsioff | Attention-based with RNA-FM | RNA-FM |
| ensirna | Ensemble + Rosetta 3D structures | - |
| gnn4sirna | Graph neural network | - |
| sirnadiscovery | Thermodynamic + k-mer features | - |

### Data Format

CSV files must contain:
- `siRNA`: 19-nt guide sequence (U notation)
- `mRNA`: 19-nt target sequence
- `efficacy`: Continuous knockdown efficacy [0,1]
- `binary`: Classification label (cell-line specific thresholds)
- `source`, `cell_line`: Provenance metadata

## Environment Variables

```bash
TORCH_HOME=/path/to/torch/cache    # PyTorch model cache
ROSETTA_DIR=/path/to/rosetta       # Required for ensirna PDB generation
```

## Notes

- All tools require GPU; Docker runs with `--gpus all`
- ENsiRNA requires Rosetta installation for PDB generation
- Feature generation requires ViennaRNA (RNAfold, RNAcofold, RNAup)
- Each tool has additional flags documented in `tools/<tool>/README.md`
