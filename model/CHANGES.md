# Model Changes

This file tracks changes made to the original model scripts.

## 2026-02-02

### Label column rename
- Training and tuning now expect `efficiency` (not `efficacy`).
- Feature builder drops `efficiency` (and no longer references `efficacy`).

### Early stopping metric control
- Added `--early-stop-metric {rmse|r2}` to `train.py`.
- R2 early stopping uses a custom metric for both XGBoost and LightGBM with maximize mode.

### Logging and warning suppression
- Training now prints progress messages for data load, feature build, training, calibration, metrics, and artifact save.
- Global warnings are suppressed in `train.py` and `inference.py`.
- Tuning suppresses warnings and moves XGBoost early-stopping to the constructor to avoid deprecation warnings.

### XGBoost / LightGBM training adjustments
- XGBoost early stopping configured in the model constructor to avoid deprecated `fit` warnings.
- LightGBM verbosity reduced to suppress split warnings.

### Reproducibility controls
- Added `--seed` and `--deterministic` to `train.py`.
- Deterministic mode keeps GPU enabled but limits threading for more repeatable runs.

### Inference updates
- Added `--use-gpu-predictor` flag to `inference.py` (CPU predictor is default).

### New utilities
- Added `metrics_from_predictions.py` to compute metrics JSON from prediction files.
