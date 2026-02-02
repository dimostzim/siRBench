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
    "srm",
    "oligoformer",
    "sirnadiscovery",
    "sirnabert",
    "attsioff",
    "gnn4sirna",
    "ensirna",
]

# 3-letter codes
TOOL_CODES = {
    "srm": "SRM",
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
# SRM (siRBench-Model) is blue, all others are gray
TOOL_COLORS = {
    "srm": "#2171B5",
    "oligoformer": "#808080",
    "sirnadiscovery": "#808080",
    "sirnabert": "#808080",
    "attsioff": "#808080",
    "gnn4sirna": "#808080",
    "ensirna": "#808080",
}

# Leftout colors: SRM is light blue, others are light gray
TOOL_LEFTOUT_COLORS = {
    "srm": "#9ECAE1",
    "oligoformer": "#C0C0C0",
    "sirnadiscovery": "#C0C0C0",
    "sirnabert": "#C0C0C0",
    "attsioff": "#C0C0C0",
    "gnn4sirna": "#C0C0C0",
    "ensirna": "#C0C0C0",
}


def load_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_txt_metrics(path):
    """Load metrics from txt file format (e.g., 'PEARSON: 0.688')."""
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
    leftout_colors = [TOOL_LEFTOUT_COLORS[t] for t in sorted_tools]

    x = np.arange(len(sorted_tools))
    width = 0.45
    bars_test = ax.bar(x - width / 2, sorted_test_values, color=test_colors,
                       edgecolor='black', linewidth=0.8, width=width, label="Test")
    bars_leftout = ax.bar(x + width / 2, sorted_leftout_values, color=leftout_colors,
                          edgecolor='black', linewidth=0.8, width=width, label="Leftout")

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

    # Add value labels on bars (2 decimal places)
    for bars, values in ((bars_test, sorted_test_values), (bars_leftout, sorted_leftout_values)):
        for bar, val in zip(bars, values):
            height = bar.get_height()
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


def render_metrics(results_dir, output_path, tools):
    # Load metrics
    test_metrics = {}
    leftout_metrics = {}
    for tool in tools:
        if tool == "srm":
            test_metrics[tool] = load_txt_metrics(
                os.path.join(results_dir, "sirbench-model", "test_metrics.txt")
            )
            leftout_metrics[tool] = load_txt_metrics(
                os.path.join(results_dir, "sirbench-model", "leftout_metrics.txt")
            )
        else:
            test_metrics[tool] = load_metrics(
                os.path.join(results_dir, tool, "metrics.json")
            )
            leftout_metrics[tool] = load_metrics(
                os.path.join(results_dir, tool, "metrics_leftout.json")
            )

    tools_with_data = [
        t for t in tools
        if test_metrics[t] is not None and leftout_metrics[t] is not None
    ]
    if not tools_with_data:
        print(f"No tools with both test/leftout metrics in {results_dir}")
        return

    fig = plt.figure(figsize=(20.5, 11.5), dpi=100)
    gs = GridSpec(2, 3, figure=fig, wspace=0.18, hspace=0.22,
                  left=0.04, right=0.98, top=0.96, bottom=0.12)

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

        if metric == "r2":
            legend_handles = [
                Patch(facecolor=TOOL_COLORS["srm"], edgecolor='black', label='Test'),
                Patch(facecolor=TOOL_LEFTOUT_COLORS["srm"], edgecolor='black', label='Leftout'),
            ]
            ax.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=13)

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

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(output_path, dpi=100, facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=300, facecolor='white', edgecolor='none')
    print(f"Saved: {pdf_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_updated_dir = os.path.join(script_dir, "..", "updated_validation_results")
    default_original_dir = os.path.join(script_dir, "..", "original_results")

    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=default_updated_dir, help="Updated results dir")
    p.add_argument("--output", default=os.path.join(default_updated_dir, "metrics_panels.png"))
    p.add_argument("--original-results-dir", default=default_original_dir)
    p.add_argument("--output-original", default=os.path.join(default_original_dir, "original_metrics_panels.png"))
    p.add_argument("--tools", nargs="+", default=TOOLS)
    args = p.parse_args()

    render_metrics(args.results_dir, args.output, args.tools)
    render_metrics(args.original_results_dir, args.output_original, args.tools)


if __name__ == "__main__":
    main()
