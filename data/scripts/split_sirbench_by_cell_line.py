#!/usr/bin/env python3
"""
Split a siRBench CSV into two CSVs based on the cell line column.

Outputs two files:
- "<prefix>_h1299_hela.csv" containing rows where cell_line is h1299 or hela
- "<prefix>_other.csv" containing all other rows

Usage:
  python3 split_sirbench_by_cell_line.py --input siRBench_full_features.csv [--out-prefix siRBench_full_features]

Notes:
- Matching is case-insensitive on the 'cell_line' column.
- The script preserves the original column order and values.
"""

import argparse
import os
import sys

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split siRBench CSV by cell_line (h1299/hela vs others)")
    p.add_argument("--input", required=True, help="Path to input siRBench CSV")
    p.add_argument(
        "--out-prefix",
        default=None,
        help=(
            "Output prefix for generated CSVs. Defaults to the input filename without extension"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"Input file not found: {in_path}", file=sys.stderr)
        return 1

    out_prefix = (
        args.out_prefix
        if args.out_prefix is not None
        else os.path.splitext(os.path.basename(in_path))[0]
    )

    try:
        df = pd.read_csv(in_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        return 1

    if "cell_line" not in df.columns:
        print("Expected column 'cell_line' not found in input CSV", file=sys.stderr)
        return 1

    # Build mask for h1299/hela (case-insensitive, trimmed)
    cell_norm = df["cell_line"].astype(str).str.strip().str.lower()
    mask_hh = cell_norm.isin({"h1299", "hela"})

    df_hh = df[mask_hh]
    df_other = df[~mask_hh]

    out_hh = f"{out_prefix}_h1299_hela.csv"
    out_other = f"{out_prefix}_other.csv"

    try:
        df_hh.to_csv(out_hh, index=False)
        df_other.to_csv(out_other, index=False)
    except Exception as e:
        print(f"Failed to write output CSVs: {e}", file=sys.stderr)
        return 1

    print(f"Wrote: {out_hh} ({len(df_hh)} rows)")
    print(f"Wrote: {out_other} ({len(df_other)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

