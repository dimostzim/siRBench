#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from metrics import format_metrics, regression_metrics, save_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-set", required=True)
    p.add_argument("--ckpt", nargs='+', required=True)
    p.add_argument("--output-csv", default="predictions.csv")
    p.add_argument("--metrics-json", default=None)
    p.add_argument("--save-dir", default="results")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--run-id", default="ensirna")
    p.add_argument("--src-root", default="ensirna_src")
    args = p.parse_args()

    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), args.src_root, "ENsiRNA"))
    run_script = os.path.join(src_root, "run.py")

    os.makedirs(args.save_dir, exist_ok=True)

    cmd = [
        sys.executable, run_script,
        "--ckpt",
    ] + args.ckpt + [
        "--test_set", args.test_set,
        "--save_dir", args.save_dir,
        "--gpu", str(args.gpu),
        "--id", args.run_id,
    ]

    subprocess.check_call(cmd, cwd=src_root)

    result_xlsx = os.path.join(args.save_dir, f"{args.run_id}_result.xlsx")
    pred_df = pd.read_excel(result_xlsx)
    pred_cols = [c for c in pred_df.columns if c.startswith('result_')]
    preds = pred_df[pred_cols].mean(axis=1).tolist() if pred_cols else []

    ids = []
    labels = []
    with open(args.test_set, 'r') as f:
        for line in f:
            item = json.loads(line)
            ids.append(item.get('id', f"row_{len(ids)}"))
            labels.append(float(item.get('efficacy', 0.0)))

    out_df = pd.DataFrame({
        "id": ids,
        "label": labels,
        "pred_label": preds,
    })
    out_df.to_csv(args.output_csv, index=False)
    metrics = regression_metrics(out_df["label"].to_numpy(), out_df["pred_label"].to_numpy())
    if args.metrics_json:
        save_metrics(metrics, args.metrics_json)
    print(format_metrics(metrics))
    print(args.output_csv)


if __name__ == "__main__":
    main()
