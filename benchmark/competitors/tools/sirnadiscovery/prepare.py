#!/usr/bin/env python
import argparse
import hashlib
import os
import shutil

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", default="data")
    p.add_argument("--dataset-name")
    p.add_argument("--id-col", default="id")
    p.add_argument("--sirna-col", default="siRNA")
    p.add_argument("--mrna-col", default="extended_mRNA")
    p.add_argument("--efficacy-col", default="efficacy")
    p.add_argument("--preprocess-dir")
    p.add_argument("--rna-ago2-dir")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    dataset_name = args.dataset_name or os.path.splitext(os.path.basename(args.input_csv))[0]

    if args.id_col not in df.columns:
        df[args.id_col] = [f"row_{i}" for i in range(len(df))]

    df[args.sirna_col] = df[args.sirna_col].astype(str).str.upper().str.replace('T', 'U')
    df[args.mrna_col] = df[args.mrna_col].astype(str).str.upper().str.replace('T', 'U')

    def stable_id(prefix, seq):
        digest = hashlib.md5(seq.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{digest}"

    df["sirna_id"] = [stable_id("sirna", seq) for seq in df[args.sirna_col]]
    df["mrna_id"] = [stable_id("mrna", seq) for seq in df[args.mrna_col]]

    out_df = pd.DataFrame({
        "id": df[args.id_col],
        "siRNA": df["sirna_id"],
        "mRNA": df["mrna_id"],
        "efficacy": df[args.efficacy_col].astype(float),
        "siRNA_seq": df[args.sirna_col],
        "mRNA_seq": df[args.mrna_col],
    })

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"{dataset_name}.csv")
    out_df.to_csv(out_csv, index=False)

    if args.preprocess_dir:
        shutil.copytree(args.preprocess_dir, os.path.join(args.output_dir, "siRNA_split_preprocess"), dirs_exist_ok=True)

    if args.rna_ago2_dir:
        shutil.copytree(args.rna_ago2_dir, os.path.join(args.output_dir, "RNA_AGO2"), dirs_exist_ok=True)

    print(out_csv)


if __name__ == "__main__":
    main()
