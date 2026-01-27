#!/usr/bin/env python
"""Generate publication-quality metrics comparison plot (1920x1080)."""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

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
    'legend.fontsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

TOOLS = [
    "oligoformer",
    "sirnadiscovery",
    "sirnabert",
    "attsioff",
    "gnn4sirna",
    "ensirna",
]

# 3-letter codes
TOOL_CODES = {
    "oligoformer": "OLI",
    "sirnadiscovery": "SDI",
    "sirnabert": "SBT",
    "attsioff": "ATT",
    "gnn4sirna": "GNN",
    "ensirna": "ENS",
}

# Metrics config: (display_name, higher_is_better)
METRICS_CONFIG = {
    "pearson": ("r", True),
    "spearman": ("$r_s$", True),
    "r2": ("R²", True),
    "mae": ("MAE", False),
    "mse": ("MSE", False),
    "rmse": ("RMSE", False),
}

# All 6 metrics in order (2 rows x 3 cols)
ALL_METRICS = ["pearson", "spearman", "r2", "mae", "mse", "rmse"]

# Colors for each tool (consistent across all panels)
TOOL_COLORS = {
    "oligoformer": "#4C72B0",
    "sirnadiscovery": "#55A868",
    "sirnabert": "#C44E52",
    "attsioff": "#8172B3",
    "gnn4sirna": "#CCB974",
    "ensirna": "#64B5CD",
}


def load_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def lighten_color(color, amount=0.4):
    """Lighten color by mixing with white."""
    r, g, b = mcolors.to_rgb(color)
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return (r, g, b)


def plot_metric_panel(ax, tools, test_values, leftout_values, metric_name, higher_is_better):
    """Plot a single metric panel with sorted bars."""
    # Sort by value
    if higher_is_better:
        # Higher is better: sort descending (best first, bars fall to lower)
        sorted_pairs = sorted(zip(tools, test_values, leftout_values), key=lambda x: x[1], reverse=True)
    else:
        # Lower is better: sort ascending (best first, bars rise to higher)
        sorted_pairs = sorted(zip(tools, test_values, leftout_values), key=lambda x: x[1], reverse=False)

    sorted_tools, sorted_test_values, sorted_leftout_values = zip(*sorted_pairs)
    test_colors = [TOOL_COLORS[t] for t in sorted_tools]
    leftout_colors = [lighten_color(TOOL_COLORS[t], amount=0.45) for t in sorted_tools]

    x = np.arange(len(sorted_tools))
    width = 0.45
    bars_test = ax.bar(x - width / 2, sorted_test_values, color=test_colors,
                       edgecolor='white', linewidth=0.8, width=width, label="Test")
    bars_leftout = ax.bar(x + width / 2, sorted_leftout_values, color=leftout_colors,
                          edgecolor='white', linewidth=0.8, width=width, label="Leftout")

    # Set axis
    ax.set_xticks(x)
    ax.set_xticklabels([TOOL_CODES[t] for t in sorted_tools])
    ax.set_title(metric_name, fontweight='bold', pad=10)

    # Y-axis limits: 0 to 10% above max
    max_val = max(max(sorted_test_values), max(sorted_leftout_values))
    min_val = min(min(sorted_test_values), min(sorted_leftout_values))

    if higher_is_better:
        ymax = max_val * 1.05
        ax.set_ylim(0, ymax)
    else:
        ymax = max_val * 1.05
        ax.set_ylim(0, ymax)

    # Add value labels on bars
    for bars, values in ((bars_test, sorted_test_values), (bars_leftout, sorted_leftout_values)):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            # Format based on magnitude
            if val < 0.01:
                fmt = f'{val:.4f}'
            elif val < 1:
                fmt = f'{val:.3f}'
            else:
                fmt = f'{val:.2f}'
            ax.annotate(
                fmt,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    return sorted_tools


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output", default="results/metrics_panels.png")
    p.add_argument("--tools", nargs="+", default=TOOLS)
    args = p.parse_args()

    # Load metrics
    test_metrics = {}
    leftout_metrics = {}
    for tool in args.tools:
        test_metrics[tool] = load_metrics(
            os.path.join(args.results_dir, tool, "metrics.json")
        )
        leftout_metrics[tool] = load_metrics(
            os.path.join(args.results_dir, tool, "metrics_leftout.json")
        )

    # Filter tools with data
    tools_with_data = [
        t for t in args.tools
        if test_metrics[t] is not None and leftout_metrics[t] is not None
    ]

    # Create figure: slightly larger to give panels more room
    fig = plt.figure(figsize=(20.5, 11.5), dpi=100)

    # Layout: 2 rows x 3 cols of equal-sized panels
    gs = GridSpec(2, 3, figure=fig, wspace=0.18, hspace=0.22,
                  left=0.04, right=0.98, top=0.96, bottom=0.12)

    # Plot all 6 metrics
    for i, metric in enumerate(ALL_METRICS):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])

        display_name, higher_is_better = METRICS_CONFIG[metric]
        test_values = [test_metrics[t].get(metric, 0) for t in tools_with_data]
        leftout_values = [
            leftout_metrics[t].get(metric, 0) if leftout_metrics[t] is not None else 0
            for t in tools_with_data
        ]

        plot_metric_panel(ax, tools_with_data, test_values, leftout_values,
                          display_name, higher_is_better)

    # Add legend outside (top right)
    tool_display_names = {
        "oligoformer": "OligoFormer",
        "sirnadiscovery": "siRNADiscovery",
        "sirnabert": "siRNABERT",
        "attsioff": "AttSiOff",
        "gnn4sirna": "GNN4siRNA",
        "ensirna": "ENsiRNA",
    }
    legend_handles = [
        Patch(facecolor=TOOL_COLORS[t], edgecolor='white', label=tool_display_names[t])
        for t in tools_with_data
    ]

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        fontsize=17,
        handlelength=1.5,
        handleheight=1.2,
        ncol=6,
    )

    # Save PNG
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(args.output, dpi=100, facecolor='white', edgecolor='none')
    print(f"Saved: {args.output}")

    # Also save PDF version
    pdf_path = args.output.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=300, facecolor='white', edgecolor='none')
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
