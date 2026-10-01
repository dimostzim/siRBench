#!/usr/bin/env python
import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from metrics import format_metrics, regression_metrics, save_metrics


def load_modules(src_root):
    sys.path.insert(0, os.path.join(src_root, "scripts"))
    from loader import data_process_loader
    from model import Oligo
    return data_process_loader, Oligo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-csv", required=True)
    p.add_argument("--data-dir")
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-csv", default="predictions.csv")
    p.add_argument("--metrics-json", default=None)
    p.add_argument("--test-name")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cuda", default="0")
    p.add_argument("--oligoformer-src", default="oligoformer_src")
    args = p.parse_args()

    data_dir = args.data_dir or os.path.dirname(os.path.abspath(args.test_csv))
    test_name = args.test_name or os.path.splitext(os.path.basename(args.test_csv))[0]

    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), args.oligoformer_src))
    data_process_loader, Oligo = load_modules(src_root)

    df = pd.read_csv(args.test_csv, dtype=str)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model = Oligo().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    loader = DataLoader(
        data_process_loader(df.index.values, df.label.values, df.y.values, df, test_name, data_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    preds = []
    with torch.no_grad():
        for batch in loader:
            siRNA, mRNA, siRNA_FM, mRNA_FM, label, _, td = batch
            siRNA = siRNA.to(device)
            mRNA = mRNA.to(device)
            siRNA_FM = siRNA_FM.to(device)
            mRNA_FM = mRNA_FM.to(device)
            td = td.to(device)
            out, _, _ = model(siRNA, mRNA, siRNA_FM, mRNA_FM, td)
            preds.extend(out[:, 1].detach().cpu().numpy().tolist())

    out_df = pd.DataFrame({
        "id": df.get("id", pd.Series([f"row_{i}" for i in range(len(df))])),
        "label": df["label"].astype(float),
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
