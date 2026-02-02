#!/usr/bin/env python
import argparse
import hashlib
import os
import shutil
import subprocess
import sys

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", default="data")
    p.add_argument("--dataset-name")
    p.add_argument("--id-col", default="id")
    p.add_argument("--sirna-col", default="siRNA")
    p.add_argument("--mrna-col", default="extended_mRNA")
    p.add_argument("--efficiency-col", default="efficiency")
    p.add_argument("--preprocess-dir")
    p.add_argument("--rna-ago2-dir")
    p.add_argument("--run-preprocess", action="store_true", help="Generate preprocess matrices with ViennaRNA.")
    p.add_argument("--sirna-len", type=int, default=19)
    p.add_argument("--mrna-len", type=int, default=57)
    p.add_argument("--sirna-svd", type=int, default=6)
    p.add_argument("--mrna-svd", type=int, default=100)
    p.add_argument("--con-svd", type=int, default=50)
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
        "efficiency": df[args.efficiency_col].astype(float),
        "siRNA_seq": df[args.sirna_col],
        "mRNA_seq": df[args.mrna_col],
    })

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"{dataset_name}.csv")
    out_df.to_csv(out_csv, index=False)

    preprocess_out = os.path.join(args.output_dir, "siRNA_split_preprocess")
    if args.preprocess_dir:
        shutil.copytree(args.preprocess_dir, preprocess_out, dirs_exist_ok=True)
    if args.run_preprocess:
        os.makedirs(os.path.join(args.output_dir, "scripts"), exist_ok=True)
        preprocess_script = os.path.join(os.path.dirname(__file__), "scripts", "preprocess.py")
        cmd = [
            sys.executable,
            preprocess_script,
            "--input-csv",
            out_csv,
            "--output-dir",
            preprocess_out,
            "--sirna-len",
            str(args.sirna_len),
            "--mrna-len",
            str(args.mrna_len),
            "--sirna-svd",
            str(args.sirna_svd),
            "--mrna-svd",
            str(args.mrna_svd),
            "--con-svd",
            str(args.con_svd),
        ]
        subprocess.check_call(cmd)

    if args.rna_ago2_dir:
        src = os.path.abspath(args.rna_ago2_dir)
        dst = os.path.abspath(os.path.join(args.output_dir, "RNA_AGO2"))
        if not os.path.isdir(src):
            raise FileNotFoundError(f"RNA_AGO2 directory not found: {src}")
        if src != dst:
            shutil.copytree(src, dst, dirs_exist_ok=True)

    print(out_csv)


if __name__ == "__main__":
    main()
