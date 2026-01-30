# siRBench

Repository layout (quick map):

- `benchmark/`: Docker-based competitor pipeline (prepare/train/test wrappers) and outputs under `benchmark/results/`.
- `data/`: CSV splits, feature columns, and helper scripts (train/val/test/leftout).
- `siRBench-model/`: Standalone model (XGBoost + LightGBM) with uv setup and training/inference scripts.
