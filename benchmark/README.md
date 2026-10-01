# Benchmark

- [predictors/](predictors/README.md): Docker setup, training and inference for the six published predictors.
- `baselines.py`: sequence scores and fitted reference baselines.
- `evaluate_predictions.py`: performance metrics, confidence intervals and paired comparisons.
- `evaluate.py`: reproduce the paper's result tables from archived predictions.

Prepare the dataset using [the data scripts](../data/scripts/README.md).
Agentomics and TabPFN training and inference are in [models/](../models/README.md).

To reproduce the result tables, run from the repository root on Linux x86-64:

```bash
uv run --locked --project data python benchmark/evaluate.py \
  --package data/archive/siRBench-v2-2026-09-27 \
  --output outputs/results
```

Use a new output directory. This evaluates saved predictions without retraining.
Each script accepts `--help` for its arguments.
