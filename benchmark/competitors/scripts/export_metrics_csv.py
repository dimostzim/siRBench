#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path

TOOLS = [
    "oligoformer",
    "sirnadiscovery",
    "sirnabert",
    "attsioff",
    "gnn4sirna",
    "ensirna",
]

METRIC_COLS = ["pearson", "spearman", "r2", "mae", "mse", "rmse", "n"]


def load_metrics_json(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def load_metrics_txt(path: Path):
    if not path.exists():
        return None
    metrics = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            if key == "r2":
                key = "r2"
            metrics[key] = float(value.strip())
    return metrics


def normalize_metrics(raw: dict):
    if raw is None:
        return None
    data = {}
    for key, val in raw.items():
        k = key.lower()
        if k == "count":
            k = "n"
        data[k] = val
    return data


def collect_rows(results_dir: Path, run_label: str):
    rows = []
    for tool in TOOLS:
        for split, fname in (("test", "metrics.json"), ("leftout", "metrics_leftout.json")):
            metrics_path = results_dir / tool / fname
            raw = load_metrics_json(metrics_path)
            data = normalize_metrics(raw)
            if data is None:
                continue
            row = {
                "run": run_label,
                "tool": tool,
                "split": split,
            }
            for col in METRIC_COLS:
                row[col] = data.get(col, "")
            rows.append(row)

    # siRBench-Model metrics live in txt files
    srm_dir = results_dir / "sirbench-model"
    for split, fname in (("test", "test_metrics.txt"), ("leftout", "leftout_metrics.txt")):
        metrics_path = srm_dir / fname
        raw = load_metrics_txt(metrics_path)
        data = normalize_metrics(raw)
        if data is None:
            continue
        row = {
            "run": run_label,
            "tool": "sirbench-model",
            "split": split,
        }
        for col in METRIC_COLS:
            row[col] = data.get(col, "")
        rows.append(row)

    return rows


def main():
    script_dir = Path(__file__).resolve().parent
    default_updated = script_dir / ".." / "updated_validation_results"
    default_original = script_dir / ".." / "original_results"
    default_output = script_dir / ".." / "metrics_all.csv"

    p = argparse.ArgumentParser()
    p.add_argument("--updated-results-dir", default=str(default_updated))
    p.add_argument("--original-results-dir", default=str(default_original))
    p.add_argument("--output", default=str(default_output))
    args = p.parse_args()

    updated_dir = Path(args.updated_results_dir).resolve()
    original_dir = Path(args.original_results_dir).resolve()
    output_path = Path(args.output).resolve()

    rows = []
    if updated_dir.exists():
        rows.extend(collect_rows(updated_dir, "updated"))
    if original_dir.exists():
        rows.extend(collect_rows(original_dir, "original"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "tool", "split"] + METRIC_COLS,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
