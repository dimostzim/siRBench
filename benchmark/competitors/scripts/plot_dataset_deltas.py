#!/usr/bin/env python
"""Plot per-tool test vs leftout metric deltas (test - leftout) in 6 panels."""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 13,
    'font.weight': 'bold',
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

TOOLS = [
    "srm",
    "oligoformer",
    "sirnadiscovery",
    "sirnabert",
    "attsioff",
    "gnn4sirna",
    "ensirna",
]

TOOL_CODES = {
    "srm": "SRM",
    "oligoformer": "OLI",
    "sirnadiscovery": "SDI",
    "sirnabert": "SBT",
    "attsioff": "ATT",
    "gnn4sirna": "GNN",
    "ensirna": "ENS",
}

METRICS_CONFIG = {
    "pearson": ("r", True),
    "spearman": ("$r_s$", True),
    "r2": ("R²", True),
    "mae": ("MAE", False),
    "mse": ("MSE", False),
    "rmse": ("RMSE", False),
}

ALL_METRICS = ["pearson", "spearman", "r2", "mae", "mse", "rmse"]

TOOL_COLORS = {
    "srm": "#2171B5",
    "oligoformer": "#808080",
    "sirnadiscovery": "#808080",
    "sirnabert": "#808080",
    "attsioff": "#808080",
    "gnn4sirna": "#808080",
    "ensirna": "#808080",
}


def load_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_txt_metrics(path):
    if not os.path.exists(path):
        return None
    metrics = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            metrics[key] = float(value.strip())
    return metrics


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results_dir = os.path.join(script_dir, "..", "updated_validation_results")

    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=default_results_dir)
    p.add_argument("--output", default=os.path.join(default_results_dir, "dataset_deltas.png"))
    p.add_argument("--tools", nargs="+", default=TOOLS)
    args = p.parse_args()

    test_metrics = {}
    leftout_metrics = {}
    for tool in args.tools:
        if tool == "srm":
            test_metrics[tool] = load_txt_metrics(
                os.path.join(args.results_dir, "sirbench-model", "test_metrics.txt")
            )
            leftout_metrics[tool] = load_txt_metrics(
                os.path.join(args.results_dir, "sirbench-model", "leftout_metrics.txt")
            )
        else:
            test_metrics[tool] = load_metrics(
                os.path.join(args.results_dir, tool, "metrics.json")
            )
            leftout_metrics[tool] = load_metrics(
                os.path.join(args.results_dir, tool, "metrics_leftout.json")
            )

    tools_with_data = [
        t for t in args.tools
        if test_metrics[t] is not None and leftout_metrics[t] is not None
    ]

    if not tools_with_data:
        raise SystemExit("No tools with both test and leftout metrics found.")

    fig = plt.figure(figsize=(20.5, 11.5), dpi=100)
    gs = GridSpec(2, 3, figure=fig, wspace=0.18, hspace=0.22,
                  left=0.04, right=0.98, top=0.96, bottom=0.12)

    for i, metric in enumerate(ALL_METRICS):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])

        display_name, _ = METRICS_CONFIG[metric]
        deltas = []
        for t in tools_with_data:
            t_val = test_metrics[t].get(metric, 0.0)
            l_val = leftout_metrics[t].get(metric, 0.0)
            deltas.append(t_val - l_val)

        x = np.arange(len(tools_with_data))
        colors = [TOOL_COLORS[t] for t in tools_with_data]
        ax.bar(x, deltas, color=colors, edgecolor='black', linewidth=0.8)

        ax.axhline(0, color='black', linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([TOOL_CODES[t] for t in tools_with_data])
        ax.set_title(f"{display_name} Δ (test - leftout)", fontweight='bold', pad=10)

        max_abs = max([abs(v) for v in deltas]) if deltas else 1.0
        ax.set_ylim(-max_abs * 1.15, max_abs * 1.15)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    # Legend text with tool codes
    tool_display_names = {
        "srm": "siRBench-Model",
        "oligoformer": "OligoFormer",
        "sirnadiscovery": "siRNADiscovery",
        "sirnabert": "siRNABERT",
        "attsioff": "AttSiOff",
        "gnn4sirna": "GNN4siRNA",
        "ensirna": "ENsiRNA",
    }
    legend_text = "   ".join(
        f"{TOOL_CODES[t]}: {tool_display_names[t]}" for t in tools_with_data
    )
    fig.text(0.5, 0.03, legend_text, ha='center', va='center', fontsize=16, fontweight='bold')

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=100, facecolor='white', edgecolor='none')
    print(f"Saved: {args.output}")
    pdf_path = args.output.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=300, facecolor='white', edgecolor='none')
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
