# Agentomics Model

This folder contains everything needed to retrain and run inference for the agentomics model (standard split).

- `model.txt`, `metrics.json`: LightGBM booster and training metrics.
- `representation.joblib`: feature pipeline (builder + scaler + feature list).
- `model_config.json`: hyperparameters.
- `build_representation.py`, `train_model.py`, `feature_builder.py`: utilities for rebuilding the representation and training a model if you place new data at `data/train.csv` and `data/validation.csv`.
- `inference.py`: CLI entry for predictions (`python -m retrained.agentomics.inference`).
- `sample_input.csv`, `sample_output.csv`: illustration of the expected input schema and output format.
- `agentomics_test_predictions.csv`, `agentomics_val_predictions.csv`: predictions on the siRBench test and validation splits.

All files live flat at this level for easy packaging.
