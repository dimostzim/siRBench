#!/usr/bin/env python
"""Generate KDE density plot for train/val/test/leftout efficacy distributions."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 13,
    'font.weight': 'bold',
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'train': '#4C72B0',
    'val': '#55A868',
    'test': '#C44E52',
    'leftout': '#8172B3',
}

LABELS = {
    'train': 'Train',
    'val': 'Validation',
    'test': 'Test',
    'leftout': 'Leftout',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/home/dtzim01/siRBench/data")
    p.add_argument("--output", default=None)
    p.add_argument("--efficacy-col", default="efficacy")
    args = p.parse_args()

    # Load datasets
    data_dir = Path(args.data_dir)
    datasets = {
        'train': data_dir / "val_split" / "siRBench_train_split.csv",
        'val': data_dir / "val_split" / "siRBench_val_split.csv",
        'test': data_dir / "siRBench_test.csv",
        'leftout': data_dir / "leftout" / "siRBench_leftout.csv",
    }

    data = {}
    for name, path in datasets.items():
        if path.exists():
            df = pd.read_csv(path)
            data[name] = df[args.efficacy_col].dropna().values
            print(f"{name}: {len(data[name])} samples")
        else:
            print(f"Warning: {path} not found")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot KDE for each dataset
    x_range = np.linspace(0, 1, 500)

    for name in ['train', 'val', 'test', 'leftout']:
        if name not in data:
            continue
        values = data[name]
        kde = stats.gaussian_kde(values)
        density = kde(x_range)
        ax.plot(x_range, density, label=LABELS[name], color=COLORS[name], linewidth=2.5)

    ax.set_xlabel("Efficacy")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)

    ax.legend(frameon=False, loc='upper right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save
    output_path = Path(args.output) if args.output else data_dir / "plots" / "efficacy_kde.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=150, facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    # Also save PDF
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300, facecolor='white', edgecolor='none')
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
