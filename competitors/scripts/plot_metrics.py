#!/usr/bin/env python
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

TOOLS = [
    "oligoformer",
    "sirnadiscovery",
    "sirnabert",
    "attsioff",
    "gnn4sirna",
    "ensirna",
]

METRICS = ["mae", "mse", "rmse", "r2", "pearson", "spearman"]


def load_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output", default="results/metrics_panels.png")
    p.add_argument("--tools", nargs="+", default=TOOLS)
    args = p.parse_args()

    tool_metrics = {}
    for tool in args.tools:
        metrics_path = os.path.join(args.results_dir, tool, "metrics.json")
        tool_metrics[tool] = load_metrics(metrics_path)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex="row", sharey="col")
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(METRICS)))
    legend_handles = [Patch(color=colors[i], label=metric) for i, metric in enumerate(METRICS)]

    for idx, tool in enumerate(args.tools):
        ax = axes[idx]
        metrics = tool_metrics[tool]
        if metrics is None:
            ax.set_title(tool)
            ax.axis("off")
            continue
        values = [metrics.get(metric) for metric in METRICS]
        heights = [v if v is not None else 0.0 for v in values]
        bars = ax.bar(METRICS, heights, color=colors)
        for bar, value in zip(bars, values):
            if value is None or not np.isfinite(value):
                bar.set_alpha(0.0)
        ax.set_title(tool)

        row = idx // 3
        col = idx % 3
        if col != 1:
            ax.tick_params(labelbottom=False)
        if row != 1:
            ax.tick_params(labelleft=False)
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[len(args.tools):]:
        ax.axis("off")

    fig.subplots_adjust(right=0.82, wspace=0.3, hspace=0.4)
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.84, 0.98),
        frameon=False,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
