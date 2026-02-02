#!/usr/bin/env python3
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a CSV into train/val with stratification."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV to split")
    parser.add_argument("--train-out", required=True, help="Output train CSV path")
    parser.add_argument("--val-out", required=True, help="Output val CSV path")
    parser.add_argument(
        "--stratify-cols",
        default="cell_line,efficiency_bin",
        help="Comma-separated columns for stratification (default: cell_line,efficiency_bin)",
    )
    parser.add_argument(
        "--efficiency-bins",
        type=int,
        default=5,
        help="Number of quantile bins for efficiency (default: 5)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Validation fraction (ignored if --val-size is set)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="Exact validation size (overrides --val-frac)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _safe_group_val_size(group_size: int, desired: int) -> int:
    if group_size <= 1:
        return 0
    return max(1, min(desired, group_size - 1))


def _compute_group_sizes(
    group_sizes: pd.Series, total_val: int
) -> List[Tuple[Tuple, int]]:
    total = int(group_sizes.sum())
    if total_val <= 0 or total_val >= total:
        raise ValueError("val size must be between 1 and total-1")

    ideal = group_sizes * (total_val / total)
    base = np.floor(ideal).astype(int)
    frac = ideal - base

    # Ensure each group keeps at least one item in train.
    for key, size in group_sizes.items():
        if size <= 1:
            base[key] = 0
        else:
            base[key] = _safe_group_val_size(size, base[key])

    remaining = total_val - int(base.sum())
    if remaining < 0:
        remaining = 0

    order = frac.sort_values(ascending=False).index.tolist()
    i = 0
    while remaining > 0 and order:
        key = order[i % len(order)]
        size = group_sizes[key]
        candidate = base[key] + 1
        if size > 1 and candidate <= size - 1:
            base[key] = candidate
            remaining -= 1
        i += 1

    return list(base.items())


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    stratify_cols = [c.strip() for c in args.stratify_cols.split(",") if c.strip()]
    if not stratify_cols:
        raise ValueError("No stratify columns provided")

    if "efficiency_bin" in stratify_cols:
        if "efficiency" not in df.columns:
            raise ValueError("efficiency column is required for efficiency_bin stratification")
        df = df.copy()
        df["efficiency_bin"] = pd.qcut(
            df["efficiency"],
            q=args.efficiency_bins,
            duplicates="drop",
        ).astype(str)

    grouped = df.groupby(stratify_cols, dropna=False)
    rng = np.random.RandomState(args.seed)

    val_indices = []
    if args.val_size is not None:
        group_sizes = grouped.size()
        sizes = _compute_group_sizes(group_sizes, args.val_size)
        for key, n in sizes:
            if n <= 0:
                continue
            group_df = grouped.get_group(key)
            val_indices.extend(
                group_df.sample(n=n, random_state=rng).index.tolist()
            )
    else:
        for _, group_df in grouped:
            n = int(round(len(group_df) * args.val_frac))
            n = _safe_group_val_size(len(group_df), n)
            if n <= 0:
                continue
            val_indices.extend(
                group_df.sample(n=n, random_state=rng).index.tolist()
            )

    val_df = df.loc[val_indices].sample(frac=1, random_state=rng).reset_index(drop=True)
    train_df = df.drop(index=val_indices).sample(frac=1, random_state=rng).reset_index(drop=True)

    train_df.to_csv(args.train_out, index=False)
    val_df.to_csv(args.val_out, index=False)


if __name__ == "__main__":
    main()
