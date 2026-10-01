#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import pandas as pd


def read_table(path, delimiter=None):
    if delimiter:
        return pd.read_csv(path, delimiter=delimiter)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\\t")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="RPISeq output table (CSV/TSV).")
    p.add_argument("--output", required=True, help="Output CSV path.")
    p.add_argument("--id-col", default=None, help="ID column name; defaults to first column.")
    p.add_argument("--rf-col", default=None, help="RF probability column name; defaults to first numeric column.")
    p.add_argument("--delimiter", default=None, help="Optional delimiter override (e.g. \\t).")
    args = p.parse_args()

    df = read_table(args.input, args.delimiter)
    if df.empty:
        raise ValueError("Input table is empty.")

    if args.id_col is None:
        id_col = df.columns[0]
    else:
        id_col = args.id_col
    if id_col not in df.columns:
        raise ValueError(f"ID column not found: {id_col}")

    if args.rf_col is None:
        rf_col = None
        for col in df.columns:
            if col == id_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                rf_col = col
                break
        if rf_col is None:
            raise ValueError("Could not infer RF probability column; pass --rf-col.")
    else:
        rf_col = args.rf_col
    if rf_col not in df.columns:
        raise ValueError(f"RF column not found: {rf_col}")

    out = df[[id_col, rf_col]].copy()
    out.columns = ["id", "RF_Classifier_prob"]
    out["id"] = out["id"].astype(str).str.lstrip(">")
    out = out.dropna(subset=["id"])
    out = out.set_index("id")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
