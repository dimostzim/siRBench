#!/usr/bin/env python
import argparse
import os
import shutil
import subprocess
import sys

import pandas as pd


def write_fasta(path, rows):
    with open(path, 'w') as f:
        for rid, seq in rows:
            f.write(f">{rid}\n{seq}\n")


def run_rnafm(rnafm_root, fasta_path, out_dir):
    workdir = os.path.join(rnafm_root, "redevelop")
    cmd = [
        sys.executable,
        "launch/predict.py",
        "--config=pretrained/extract_embedding.yml",
        f"--data_path={fasta_path}",
        f"--save_dir={out_dir}",
        "--save_frequency", "1",
        "--save_embeddings",
    ]
    subprocess.check_call(cmd, cwd=workdir)


def _has_embeddings(root):
    if not os.path.isdir(root):
        return False
    rep_dir = os.path.join(root, "representations")
    if os.path.isdir(rep_dir):
        for name in os.listdir(rep_dir):
            if name.endswith(".npy"):
                return True
    for name in os.listdir(root):
        if name.endswith(".npy"):
            return True
    return False


def rnafm_ready(output_dir):
    sirna_dir = os.path.join(output_dir, "data", "RNAFM_sirna")
    mrna_dir = os.path.join(output_dir, "data", "RNAFM_mrna")
    return _has_embeddings(sirna_dir) and _has_embeddings(mrna_dir)


def flatten_rnafm(root):
    rep_dir = os.path.join(root, "representations")
    if not os.path.isdir(rep_dir):
        return
    for name in os.listdir(rep_dir):
        if not name.endswith(".npy"):
            continue
        src = os.path.abspath(os.path.join(rep_dir, name))
        dst = os.path.join(root, name)
        if os.path.exists(dst):
            continue
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", default="data")
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--id-col", default="id")
    p.add_argument("--sirna-col", default="siRNA")
    p.add_argument("--mrna-col", default="extended_mRNA")
    p.add_argument("--efficiency-col", default="efficiency")
    p.add_argument("--source-col", default="source")
    p.add_argument("--biopred-col", default="s-Biopredsi")
    p.add_argument("--dsir-col", default="DSIR")
    p.add_argument("--iscore-col", default="i-score")
    p.add_argument("--run-rnafm", action="store_true")
    p.add_argument("--rnafm-root", default="attsioff_src/RNA-FM")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        base = os.path.basename(args.input_csv)
        dataset_name = os.path.splitext(base)[0]
    if args.id_col not in df.columns:
        df[args.id_col] = [f"row_{i}" for i in range(len(df))]

    df[args.sirna_col] = df[args.sirna_col].astype(str).str.upper().str.replace('T', 'U')
    df[args.mrna_col] = df[args.mrna_col].astype(str).str.upper().str.replace('T', 'U')

    def ensure_col(col, default=0.0):
        if col not in df.columns:
            df[col] = default

    ensure_col(args.biopred_col, 0.0)
    ensure_col(args.dsir_col, 0.0)
    ensure_col(args.iscore_col, 0.0)

    df["RNAFM_ind"] = list(range(len(df)))

    out_df = pd.DataFrame({
        "Antisense": df[args.sirna_col],
        "mrna": df[args.mrna_col],
        "s-Biopredsi": df[args.biopred_col],
        "DSIR": df[args.dsir_col],
        "i-score": df[args.iscore_col],
        "inhibition": df[args.efficiency_col],
        "RNAFM_ind": df["RNAFM_ind"],
        "source_paper": df[args.source_col] if args.source_col in df.columns else "NA",
        "id": df[args.id_col],
    })

    os.makedirs(args.output_dir, exist_ok=True)
    data_root = os.path.join(args.output_dir, "data")
    os.makedirs(data_root, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"{dataset_name}.csv")
    out_df.to_csv(out_csv, index=False)

    if args.run_rnafm or not rnafm_ready(args.output_dir):
        rnafm_root = os.path.abspath(args.rnafm_root)
        if not os.path.isdir(rnafm_root):
            raise FileNotFoundError(f"RNA-FM not found: {rnafm_root}")

        rnafm_sirna = os.path.join(data_root, "RNAFM_sirna")
        rnafm_mrna = os.path.join(data_root, "RNAFM_mrna")
        os.makedirs(rnafm_sirna, exist_ok=True)
        os.makedirs(rnafm_mrna, exist_ok=True)

        sirna_fa = os.path.join(data_root, "sirna.fa")
        mrna_fa = os.path.join(data_root, "mrna.fa")

        write_fasta(sirna_fa, [(f"{i:04d}", s) for i, s in enumerate(out_df["Antisense"])])
        write_fasta(mrna_fa, [(f"{i:04d}", s) for i, s in enumerate(out_df["mrna"])])

        run_rnafm(rnafm_root, sirna_fa, rnafm_sirna)
        run_rnafm(rnafm_root, mrna_fa, rnafm_mrna)

    rnafm_sirna = os.path.join(data_root, "RNAFM_sirna")
    rnafm_mrna = os.path.join(data_root, "RNAFM_mrna")
    if os.path.isdir(rnafm_sirna):
        flatten_rnafm(rnafm_sirna)
    if os.path.isdir(rnafm_mrna):
        flatten_rnafm(rnafm_mrna)

    print(out_csv)


if __name__ == "__main__":
    main()
