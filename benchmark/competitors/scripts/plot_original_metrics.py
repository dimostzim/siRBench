#!/usr/bin/env python
"""Generate 6-panel test vs leftout metrics plot for original results."""
import argparse
import os

from plot_metrics import TOOLS, render_metrics


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_original_dir = os.path.join(script_dir, "..", "original_results")

    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=default_original_dir)
    p.add_argument("--output", default=os.path.join(default_original_dir, "original_metrics_panels.png"))
    p.add_argument("--tools", nargs="+", default=TOOLS)
    args = p.parse_args()

    render_metrics(args.results_dir, args.output, args.tools)


if __name__ == "__main__":
    main()
