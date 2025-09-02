#!/usr/bin/env python3
import argparse
import csv
import math
import os
import random
from collections import defaultdict, Counter


def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames, rows


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rank_bins(values, n_bins):
    # Assign bin indices 0..n_bins-1 by rank-based quantiles
    if not values:
        return []
    # values is list of (idx, value)
    sorted_vals = sorted(values, key=lambda x: (float(x[1]), x[0]))
    n = len(sorted_vals)
    bins = {}
    for rank, (idx, v) in enumerate(sorted_vals):
        # rank in [0, n-1], bin = floor(rank / n * n_bins)
        # guard edge case rank == n -> not possible; ensure max bin < n_bins
        b = int(math.floor((rank / max(1, n)) * n_bins))
        if b >= n_bins:
            b = n_bins - 1
        bins[idx] = b
    return bins


def summarize(rows, key_fields):
    c = Counter()
    for r in rows:
        key = tuple(r[k] for k in key_fields)
        c[key] += 1
    return c


def main():
    ap = argparse.ArgumentParser(description="Stratified 90/10 split by cell_line, binary, and binned efficacy")
    ap.add_argument("--input", default="data/siRBench_train.csv", help="Input CSV path")
    ap.add_argument("--train_out", default="data/siRBench_train_90.csv", help="Output train CSV path")
    ap.add_argument("--val_out", default="data/siRBench_val_10.csv", help="Output validation CSV path")
    ap.add_argument("--test_size", type=float, default=0.10, help="Validation fraction (default 0.10)")
    ap.add_argument("--bins", type=int, default=5, help="Number of efficacy bins per cell line (default 5)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = ap.parse_args()

    fieldnames, rows = read_csv(args.input)

    # Validate required columns
    for col in ("cell_line", "binary", "efficacy"):
        if col not in fieldnames:
            raise SystemExit(f"Missing required column '{col}' in {args.input}")

    # Assign efficacy bins per cell line using rank-based quantiles
    idx_to_bin = {}
    by_cell = defaultdict(list)
    for i, r in enumerate(rows):
        by_cell[r["cell_line"]].append((i, r["efficacy"]))
    for cell, val_list in by_cell.items():
        bins = rank_bins(val_list, args.bins)
        idx_to_bin.update({i: b for i, b in bins.items()})

    # Build stratification keys: cell_line | binary | eff_bin
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        eff_bin = idx_to_bin[i]
        key = (r["cell_line"], r["binary"], str(eff_bin))
        groups[key].append(i)

    # Split within each group
    random.seed(args.seed)
    train_idx = []
    val_idx = []
    small_groups = []
    for key, idxs in groups.items():
        random.shuffle(idxs)
        n = len(idxs)
        # target count for val
        target_val = int(round(args.test_size * n))
        if n == 1:
            # too small to split; keep in train
            small_groups.append((key, n))
            target_val = 0
        elif target_val < 1:
            target_val = 1
        elif target_val >= n:
            target_val = n - 1
        val_idx.extend(idxs[:target_val])
        train_idx.extend(idxs[target_val:])

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]

    write_csv(args.train_out, fieldnames, train_rows)
    write_csv(args.val_out, fieldnames, val_rows)

    # Print concise summaries
    total = len(rows)
    print(f"Total: {total}")
    print(f"Train: {len(train_rows)} ({len(train_rows)/total:.1%}) -> {args.train_out}")
    print(f"Val:   {len(val_rows)} ({len(val_rows)/total:.1%}) -> {args.val_out}")

    def pretty(counter, title):
        print(title)
        for k, v in sorted(counter.items()):
            print(f"  {k}: {v}")

    # Summaries by cell_line and binary
    pretty(summarize(train_rows, ["cell_line", "binary"]), "Train counts by (cell_line, binary):")
    pretty(summarize(val_rows, ["cell_line", "binary"]), "Val counts by (cell_line, binary):")

    if small_groups:
        print("Note: Some tiny groups (cell_line, binary, eff_bin) had size 1 and were kept entirely in train:")
        for key, n in small_groups:
            print(f"  {key}: {n}")


if __name__ == "__main__":
    main()

