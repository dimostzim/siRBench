import json
import math

import numpy as np


def _to_1d_array(values):
    return np.asarray(values, dtype=float).reshape(-1)


def _filter_finite(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def _safe_corr(x, y):
    if len(x) < 2:
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)

    sorted_vals = values[order]
    start = 0
    while start < len(sorted_vals):
        end = start + 1
        while end < len(sorted_vals) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        if end - start > 1:
            avg_rank = np.mean(ranks[order[start:end]])
            ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _clean(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return float(value)


def regression_metrics(y_true, y_pred):
    y_true = _to_1d_array(y_true)
    y_pred = _to_1d_array(y_pred)
    y_true, y_pred = _filter_finite(y_true, y_pred)

    n = int(len(y_true))
    metrics = {"n": n}
    if n == 0:
        metrics.update({
            "mae": None,
            "mse": None,
            "rmse": None,
            "r2": None,
            "pearson": None,
            "spearman": None,
        })
        return metrics

    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(err)))

    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = None
    if denom > 0:
        r2 = float(1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom)

    pearson = _safe_corr(y_true, y_pred)
    spearman = _safe_corr(_rankdata(y_true), _rankdata(y_pred))

    metrics.update({
        "mae": _clean(mae),
        "mse": _clean(mse),
        "rmse": _clean(rmse),
        "r2": _clean(r2),
        "pearson": _clean(pearson),
        "spearman": _clean(spearman),
    })
    return metrics


def format_metrics(metrics):
    return json.dumps(metrics, sort_keys=True)


def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
