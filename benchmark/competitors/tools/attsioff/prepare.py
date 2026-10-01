#!/usr/bin/env python
import argparse
import hashlib
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


def embedding_path(root, identifier):
    for directory in (root, os.path.join(root, "representations")):
        path = os.path.join(directory, f"{identifier}.npy")
        if os.path.isfile(path):
            return path
    return None


def extract_missing(rnafm_root, rows, out_dir, fasta_path, force=False):
    missing = [(identifier, sequence) for identifier, sequence in rows
               if force or embedding_path(out_dir, identifier) is None]
    if missing:
        write_fasta(fasta_path, missing)
        run_rnafm(rnafm_root, fasta_path, out_dir)
    absent = [identifier for identifier, _ in rows
              if embedding_path(out_dir, identifier) is None]
    if absent:
        raise FileNotFoundError(f"RNA-FM did not produce {len(absent)} embeddings; first: {absent[0]}")


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
            os.symlink(os.path.relpath(src, root), dst)
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
        df[args.id_col] = df["record_id"] if "record_id" in df.columns else [f"row_{i}" for i in range(len(df))]

    df[args.sirna_col] = df[args.sirna_col].astype(str).str.upper().str.replace('T', 'U')
    df[args.mrna_col] = df[args.mrna_col].astype(str).str.upper().str.replace('T', 'U')

    def ensure_col(col, default=0.0):
        if col not in df.columns:
            df[col] = default

    ensure_col(args.biopred_col, 0.0)
    ensure_col(args.dsir_col, 0.0)
    ensure_col(args.iscore_col, 0.0)

    df["RNAFM_ind"] = [
        "pair_" + hashlib.sha256(f"{guide}|{context}".encode()).hexdigest()
        for guide, context in zip(df[args.sirna_col], df[args.mrna_col])
    ]

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

    rnafm_root = os.path.abspath(args.rnafm_root)
    if not os.path.isdir(rnafm_root):
        raise FileNotFoundError(f"RNA-FM not found: {rnafm_root}")
    for sequence_column, subdirectory in (("Antisense", "RNAFM_sirna"), ("mrna", "RNAFM_mrna")):
        embedding_dir = os.path.join(data_root, subdirectory)
        os.makedirs(embedding_dir, exist_ok=True)
        rows = list(out_df[["RNAFM_ind", sequence_column]].drop_duplicates().itertuples(index=False, name=None))
        fasta_path = os.path.join(data_root, f"{dataset_name}_{subdirectory}.fa")
        extract_missing(rnafm_root, rows, embedding_dir, fasta_path, args.run_rnafm)

    rnafm_sirna = os.path.join(data_root, "RNAFM_sirna")
    rnafm_mrna = os.path.join(data_root, "RNAFM_mrna")
    if os.path.isdir(rnafm_sirna):
        flatten_rnafm(rnafm_sirna)
    if os.path.isdir(rnafm_mrna):
        flatten_rnafm(rnafm_mrna)

    print(out_csv)


if __name__ == "__main__":
    main()
