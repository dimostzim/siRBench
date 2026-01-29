# AGENT.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

siRBench is a benchmark framework for evaluating computational methods for predicting siRNA (small interfering RNA) knockdown efficacy. It provides:
- A curated dataset of 4,098 siRNA-target pairs across 7 human cell lines
- Standardized train/val/test CSV splits plus an additional leftout set for generalization checks
- Docker-based wrappers for 6 different prediction methods with unified interfaces
- Thermodynamic feature calculations using ViennaRNA utilities
- A standalone non-Docker baseline model under `siRBench-model/` (XGBoost + LightGBM with calibration)

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
./competitors/run_tool.sh --tool oligoformer

# Or with custom paths:
DATA_ROOT=/path/to/data ./competitors/run_tool.sh --tool oligoformer
```

### Individual Steps
```bash
# Prepare data (converts to tool-specific format)
python3 competitors/prepare.py --tool oligoformer \
  --input-csv data/siRBench_train.csv \
  --output-dir competitors/data/oligoformer \
  --dataset-name train

# Train model
python3 competitors/scripts/train.py --tool oligoformer \
  --train-csv competitors/data/oligoformer/train.csv \
  --val-csv competitors/data/oligoformer/val.csv \
  --data-dir competitors/data/oligoformer \
  --model-dir competitors/models/oligoformer

# Test and evaluate
python3 competitors/scripts/test.py --tool oligoformer \
  --test-csv competitors/data/oligoformer/test.csv \
  --data-dir competitors/data/oligoformer \
  --model-path competitors/models/oligoformer/model.pt \
  --output-csv competitors/preds.csv \
  --metrics-json competitors/metrics.json
```

### Leftout Evaluation
```bash
# Evaluate an additional unseen set
./competitors/run_tool.sh --tool oligoformer --leftout data/siRBench_leftout.csv
```

### siRBench-model (non-Docker baseline)
```bash
# Train (uses numeric_label column)
python3 siRBench-model/train.py \
  --train-data data/siRBench_train.csv \
  --validation-data data/siRBench_val_split.csv \
  --artifacts-dir siRBench-model/training_artifacts

# Inference
python3 siRBench-model/inference.py \
  --input data/siRBench_test.csv \
  --output siRBench-model/preds.csv \
  --artifacts-dir siRBench-model/training_artifacts
```

### Feature Generation (requires ViennaRNA)
```bash
cd data/scripts
python3 make_all_features.py ../siRBench_full_base.csv -o ../siRBench_with_features.csv
```

## Architecture

### Directory Structure
```
siRBench/
├── data/                           # Datasets and feature utilities
│   ├── siRBench_train.csv          # Train split
│   ├── siRBench_val_split.csv      # Validation split
│   ├── siRBench_test.csv           # Test split
│   ├── siRBench_leftout.csv        # Leftout holdout set
│   ├── thresholds*.csv/md          # Cell-line thresholds for binary labels
│   └── scripts/                    # ViennaRNA feature calculators + split utilities
└── competitors/                    # Unified ML pipeline
    ├── prepare.py                  # Wrapper → tools/<tool>/prepare.py
    ├── scripts/                    # Wrapper utilities + docker runner
    │   ├── runner.py               # Docker orchestration (path translation, GPU)
    │   ├── train.py                # Wrapper → tools/<tool>/train.py
    │   ├── test.py                 # Wrapper → tools/<tool>/test.py
    │   └── metrics.py              # Evaluation (MAE, MSE, RMSE, R², Pearson, Spearman)
    └── tools/<tool>/               # Tool-specific implementations
└── siRBench-model/                 # Standalone baseline model (XGB+LGBM)
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

## Results and Reports

- `results/<tool>/preds.csv` and `results/<tool>/metrics.json` contain per-tool predictions/metrics.
- Root-level `leftout_metrics.txt` and `test_metrics.txt` capture summary metrics for the benchmark splits.
- Plots are saved to `results/metrics_panels.png`, `results/efficacy_kde.png`, and `data/siRBench_train_val_split.png`.

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
- See `competitors/README.md`, `competitors/tools_overview.md`, and `competitors/TOOLS_CHANGES.md` for tool-specific notes
