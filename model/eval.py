#!/usr/bin/env python3
import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from inference import load_artifacts, predict

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pearson = float(stats.pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
    spearman = float(stats.spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "count": int(len(y_true)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run inference + metrics for test and leftout splits.")
    parser.add_argument("--artifacts-dir", default="training_artifacts", help="Artifacts directory")
    parser.add_argument("--repo-root", default=None, help="Repo root (defaults to inferred)")
    parser.add_argument("--output-dir", default=".", help="Directory to write predictions/metrics")
    parser.add_argument("--use-gpu-predictor", action="store_true", help="Use XGBoost GPU predictor for inference")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = Path(args.repo_root) if args.repo_root else script_dir.parent
    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = (script_dir / artifacts_dir).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (script_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "test": repo_root / "data" / "siRBench_test.csv",
        "leftout": repo_root / "data" / "siRBench_leftout.csv",
    }

    encoder, numeric_cols, _, xgb_model, lgb_model, calibrator = load_artifacts(
        artifacts_dir,
        use_gpu_predictor=args.use_gpu_predictor,
    )

    for name, data_csv in splits.items():
        preds_csv = output_dir / f"eval_predictions_{name}.csv"
        metrics_json = output_dir / f"metrics_{name}.json"
        print(f"[eval] {name}: running inference")
        df = pd.read_csv(data_csv)
        preds = predict(df, encoder, numeric_cols, xgb_model, lgb_model, calibrator)

        out_df = pd.DataFrame({"prediction": preds})
        if "id" in df.columns:
            out_df.insert(0, "id", df["id"])
        out_df.to_csv(preds_csv, index=False)
        print(f"[eval] {name}: saved predictions -> {preds_csv}")

        if "efficiency" not in df.columns:
            raise ValueError(f"Missing efficiency column in {data_csv}")
        metrics = compute_metrics(df["efficiency"].to_numpy(), preds)
        metrics_json.write_text(json.dumps(metrics, indent=2))
        print(f"[eval] {name}: metrics -> {metrics_json}")


if __name__ == "__main__":
    main()
