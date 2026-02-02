#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pandas as pd


def write_fasta(path, rows):
    with open(path, "w") as f:
        for rid, seq in rows:
            f.write(f">{rid}\n{seq}\n")


def chunk_rows(rows, size):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True, help="Prepared CSV with siRNA/mRNA ids and sequences.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--type", required=True, choices=["sirna", "mrna"])
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--sirna-id-col", default="siRNA")
    p.add_argument("--mrna-id-col", default="mRNA")
    p.add_argument("--sirna-seq-col", default="siRNA_seq")
    p.add_argument("--mrna-seq-col", default="mRNA_seq")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.type == "sirna":
        id_col = args.sirna_id_col
        seq_col = args.sirna_seq_col
    else:
        id_col = args.mrna_id_col
        seq_col = args.mrna_seq_col

    if id_col not in df.columns or seq_col not in df.columns:
        raise ValueError(f"Missing columns in {args.input_csv}: {id_col}, {seq_col}")

    rows = (
        df[[id_col, seq_col]]
        .drop_duplicates()
        .assign(**{seq_col: lambda d: d[seq_col].astype(str).str.upper().str.replace("T", "U")})
    )
    items = list(rows.itertuples(index=False, name=None))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = "siRNA" if args.type == "sirna" else "mRNA"
    for idx, chunk in enumerate(chunk_rows(items, args.batch_size), start=1):
        out_path = out_dir / f"{prefix}_batch_{idx:03d}.fasta"
        write_fasta(out_path, chunk)

    print(out_dir)


if __name__ == "__main__":
    main()
