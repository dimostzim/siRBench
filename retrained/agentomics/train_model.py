import json
import math
import random
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "model_config.json"
REPRESENTATION_PATH = BASE_DIR / "representation.joblib"
MODEL_PATH = BASE_DIR / "model.txt"
METRICS_PATH = BASE_DIR / "train_metrics.json"


def compute_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_lightgbm(X_train, y_train, X_val, y_val, params, num_boost_round, early_stopping_rounds):
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    callbacks = [lgb.log_evaluation(period=0), lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    return booster


def main():
    config = json.loads(CONFIG_PATH.read_text())
    data = joblib.load(REPRESENTATION_PATH)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    seed = config.get("random_seed", 42)
    np.random.seed(seed)
    random.seed(seed)

    lgb_params = dict(config["lightgbm"])
    num_boost_round = lgb_params.pop("n_estimators", 1000)
    early_stopping_rounds = lgb_params.pop("early_stopping_rounds", 200)
    device_type = lgb_params.get("device_type", "cpu")
    lgb_params.setdefault("seed", seed)
    lgb_params.setdefault("feature_fraction_seed", seed)
    lgb_params.setdefault("bagging_seed", seed)

    booster = None
    used_device = device_type
    try:
        booster = train_lightgbm(X_train, y_train, X_val, y_val, lgb_params, num_boost_round, early_stopping_rounds)
    except Exception:
        if device_type.lower() == "gpu":
            lgb_params["device_type"] = "cpu"
            used_device = "cpu"
            booster = train_lightgbm(X_train, y_train, X_val, y_val, lgb_params, num_boost_round, early_stopping_rounds)
        else:
            raise

    best_iteration = booster.best_iteration or num_boost_round
    train_pred = booster.predict(X_train, num_iteration=best_iteration)
    val_pred = booster.predict(X_val, num_iteration=best_iteration)

    train_metrics = compute_metrics(y_train, train_pred)
    val_metrics = compute_metrics(y_val, val_pred)

    booster.save_model(str(MODEL_PATH), num_iteration=best_iteration)

    results = {
        "best_iteration": best_iteration,
        "device_used": used_device,
        "best_scores": booster.best_score,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    METRICS_PATH.write_text(json.dumps(results, indent=2))

    summary = {
        "best_iteration": best_iteration,
        "device": used_device,
        "train_rmse": round(train_metrics["rmse"], 4),
        "val_rmse": round(val_metrics["rmse"], 4),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
