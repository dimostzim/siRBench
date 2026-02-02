#!/usr/bin/env python
import argparse
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", default="data")
    p.add_argument("--dataset-name")
    p.add_argument("--id-col", default="id")
    p.add_argument("--sirna-col", default="siRNA")
    p.add_argument("--efficiency-col", default="efficiency")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    dataset_name = args.dataset_name or os.path.splitext(os.path.basename(args.input_csv))[0]

    if args.id_col not in df.columns:
        df[args.id_col] = [f"row_{i}" for i in range(len(df))]

    out_df = pd.DataFrame({
        "id": df[args.id_col],
        "siRNA": df[args.sirna_col].astype(str).str.upper().str.replace('T', 'U'),
        "efficiency": df[args.efficiency_col].astype(float),
    })

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"{dataset_name}.csv")
    out_df.to_csv(out_csv, index=False)
    print(out_csv)


if __name__ == "__main__":
    main()
