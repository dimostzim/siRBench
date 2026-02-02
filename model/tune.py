#!/usr/bin/env python3
import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from feature_builder import build_feature_matrix

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")


def compute_metrics(y_true, y_pred):
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
    }


def log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_params(rng, n_jobs):
    xgb_params = dict(
        booster="dart",
        n_estimators=int(rng.integers(2000, 3201)),
        learning_rate=log_uniform(rng, 0.02, 0.04),
        max_depth=int(rng.integers(6, 9)),
        min_child_weight=log_uniform(rng, 0.8, 2.0),
        subsample=float(rng.uniform(0.8, 1.0)),
        colsample_bytree=float(rng.uniform(0.8, 1.0)),
        gamma=float(rng.uniform(0.0, 0.2)),
        reg_alpha=float(rng.uniform(0.0, 0.2)),
        reg_lambda=float(rng.uniform(0.7, 1.5)),
        rate_drop=float(rng.uniform(0.05, 0.2)),
        skip_drop=float(rng.uniform(0.3, 0.7)),
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=n_jobs,
        random_state=42,
        early_stopping_rounds=250,
    )
    lgb_params = dict(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=int(rng.integers(5000, 9001)),
        learning_rate=log_uniform(rng, 0.007, 0.02),
        max_depth=int(rng.integers(8, 11)),
        num_leaves=int(rng.integers(256, 1025)),
        min_child_samples=int(rng.integers(5, 21)),
        min_child_weight=log_uniform(rng, 1e-3, 5e-2),
        subsample=float(rng.uniform(0.8, 0.95)),
        subsample_freq=1,
        colsample_bytree=float(rng.uniform(0.8, 0.95)),
        reg_alpha=float(rng.uniform(0.0, 0.2)),
        reg_lambda=float(rng.uniform(0.5, 1.2)),
        random_state=42,
        n_jobs=n_jobs,
        verbosity=-1,
    )
    ens_weight = float(rng.uniform(0.3, 0.7))
    return xgb_params, lgb_params, ens_weight


def fit_xgb(xgb_params, X_train, y_train, X_val, y_val, w_train, w_val, use_gpu):
    params = dict(xgb_params)
    if use_gpu:
        params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
    else:
        params.update({"tree_method": "hist", "predictor": "auto"})

    try:
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            verbose=False,
        )
        return model
    except Exception:
        params.update({"tree_method": "hist", "predictor": "auto"})
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            verbose=False,
        )
        return model


def fit_lgb(lgb_params, X_train, y_train, X_val, y_val, w_train, w_val, use_gpu):
    params = dict(lgb_params)
    params["device"] = "gpu" if use_gpu else "cpu"
    callbacks = [lgb.early_stopping(500, verbose=False)]
    try:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            eval_metric="rmse",
            callbacks=callbacks,
        )
        return model
    except Exception:
        params["device"] = "cpu"
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            eval_metric="rmse",
            callbacks=callbacks,
        )
        return model


def run_trial(trial_id, params, metric, X_train, y_train, X_val, y_val, w_train, w_val, use_gpu):
    xgb_params, lgb_params, ens_weight = params

    xgb_model = fit_xgb(xgb_params, X_train, y_train, X_val, y_val, w_train, w_val, use_gpu)
    lgb_model = fit_lgb(lgb_params, X_train, y_train, X_val, y_val, w_train, w_val, use_gpu)

    xgb_pred = xgb_model.predict(X_val)
    lgb_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)
    avg_pred = ens_weight * xgb_pred + (1.0 - ens_weight) * lgb_pred

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(avg_pred, y_val)
    calibrated_pred = np.clip(calibrator.predict(avg_pred), 0.0, 1.0)

    metrics = compute_metrics(y_val, calibrated_pred)
    score = metrics[metric]

    row = {
        "trial": trial_id,
        "metric": metric,
        "score": score,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "pearson": metrics["pearson"],
        "spearman": metrics["spearman"],
        "xgb_best_iteration": int(getattr(xgb_model, "best_iteration", None) or xgb_model.get_booster().best_ntree_limit),
        "lgb_best_iteration": int(lgb_model.best_iteration_ if lgb_model.best_iteration_ is not None else lgb_model.n_estimators),
        "ensemble_weight": ens_weight,
        "xgb_params": xgb_params,
        "lgb_params": lgb_params,
    }
    return row


def set_seed(seed):
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="tuning_runs")
    parser.add_argument("--metric", choices=["r2", "rmse"], default="r2")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--deterministic", action="store_true", help="Best-effort determinism; mainly affects RNG seeds.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_data)
    val_df = pd.read_csv(args.validation_data)

    # Build features once for all trials.
    X_train, feature_names, encoder = build_feature_matrix(
        train_df,
        fit_encoder=True,
        artifacts_path=os.path.join(out_dir, "feature_artifacts.json"),
    )
    X_val, _, _ = build_feature_matrix(val_df, encoder=encoder, fit_encoder=False)

    y_train = train_df["efficiency"].to_numpy(dtype=np.float32)
    y_val = val_df["efficiency"].to_numpy(dtype=np.float32)
    w_train = 1.0 + 3.5 * np.abs(y_train - 0.5)
    w_val = 1.0 + 3.5 * np.abs(y_val - 0.5)

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    worker_n_jobs = -1 if args.jobs <= 1 else 1
    params_list = [sample_params(rng, worker_n_jobs) for _ in range(args.trials)]

    if args.device == "auto":
        use_gpu = args.jobs <= 1
    else:
        use_gpu = args.device == "gpu"

    results = []
    if args.jobs <= 1:
        best = None
        best_score = None
        for trial in range(1, args.trials + 1):
            row = run_trial(trial, params_list[trial - 1], args.metric, X_train, y_train, X_val, y_val, w_train, w_val, use_gpu)
            results.append(row)
            score = row["score"]
            if best_score is None:
                best_score = score
                best = row
            else:
                improved = score > best_score if args.metric == "r2" else score < best_score
                if improved:
                    best_score = score
                    best = row

            print(f"trial={trial:03d} {args.metric}={score:.5f} r2={row['r2']:.5f} rmse={row['rmse']:.5f}")

            pd.DataFrame(results).to_json(out_dir / "trials.json", orient="records", indent=2)
            with open(out_dir / "best_params.json", "w") as f:
                json.dump(best, f, indent=2)
    else:
        rows = joblib.Parallel(n_jobs=args.jobs, max_nbytes="256M")(
            joblib.delayed(run_trial)(
                idx + 1,
                params_list[idx],
                args.metric,
                X_train,
                y_train,
                X_val,
                y_val,
                w_train,
                w_val,
                use_gpu,
            )
            for idx in range(args.trials)
        )
        results = sorted(rows, key=lambda r: r["trial"])
        if args.metric == "r2":
            best = max(results, key=lambda r: r["score"])
        else:
            best = min(results, key=lambda r: r["score"])

        pd.DataFrame(results).to_json(out_dir / "trials.json", orient="records", indent=2)
        with open(out_dir / "best_params.json", "w") as f:
            json.dump(best, f, indent=2)

        for row in results:
            print(f"trial={row['trial']:03d} {args.metric}={row['score']:.5f} r2={row['r2']:.5f} rmse={row['rmse']:.5f}")

    print(f"best_{args.metric}={best['score']:.5f}")
    print(f"results saved to {out_dir}")


if __name__ == "__main__":
    main()
