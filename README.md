# siRBench

Repository layout (quick map):

- `benchmark/`: Docker-based competitor pipeline (prepare/train/test wrappers). Outputs go to `benchmark/competitors/updated_validation_results/` by default, or `benchmark/competitors/original_results/` when running with `--original`.
- `data/`: CSV splits, feature columns, and helper scripts (train/val/test/leftout).
- `siRBench-model/`: Standalone model (XGBoost + LightGBM) with uv setup and training/inference scripts.
