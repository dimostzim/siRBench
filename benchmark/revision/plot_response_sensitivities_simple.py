"""Plot all four verified input-extent controls without modifying scientific results."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COMPARISONS = (
    ("attsioff_input_extent", "AttSiOff", "21/59-nt input"),
    ("sirnadiscovery_input_extent", "siRNADiscovery", "Archived target extent"),
    ("gnn4sirna_input_extent", "GNN4siRNA", "Archived target extent"),
    ("ensirna_input_extent", "ENsiRNA", "61-nt target context"),
)
METRICS = (("pearson_r", "Pearson r"), ("r2", "R²"))
COHORTS = (("test", "Grouped test", "#176B9A"),
           ("hela", "Eligible HeLa", "#B65A22"))


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_results(summary_rows, seed_rows):
    selected = {}
    for comparison, _, _ in COMPARISONS:
        for cohort, _, _ in COHORTS:
            pairs = sorted(
                (row for row in seed_rows
                 if row["comparison"] == comparison and row["cohort"] == cohort),
                key=lambda row: int(row["training_seed"]),
            )
            assert [int(row["training_seed"]) for row in pairs] == [0, 1, 2]
            for metric, _ in METRICS:
                summaries = [row for row in summary_rows
                             if (row["comparison"], row["cohort"], row["metric"])
                             == (comparison, cohort, metric)]
                assert len(summaries) == 1
                summary = summaries[0]
                assert summary["status"] == "complete"
                assert int(summary["paired_seeds"]) == 3
                assert all(row["n"] == summary["n"] for row in pairs)
                for prefix in ("reference", "variant", "delta"):
                    values = [float(row[f"{prefix}_{metric}"]) for row in pairs]
                    assert all(math.isfinite(value) for value in values)
                    assert math.isclose(statistics.mean(values),
                                        float(summary[f"{prefix}_mean"]),
                                        rel_tol=0, abs_tol=1e-12)
                    assert math.isclose(statistics.stdev(values),
                                        float(summary[f"{prefix}_sd"]),
                                        rel_tol=0, abs_tol=1e-12)
                for row in pairs:
                    assert math.isclose(float(row[f"variant_{metric}"])
                                        - float(row[f"reference_{metric}"]),
                                        float(row[f"delta_{metric}"]),
                                        rel_tol=0, abs_tol=1e-12)
                selected[comparison, cohort, metric] = (summary, pairs)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Choose a new, empty figure directory')
    summary_path = args.input_dir / "paired_summary.csv"
    seeds_path = args.input_dir / "per_seed_metrics.csv"
    selected = validate_results(read_rows(summary_path), read_rows(seeds_path))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 12.5, "axes.labelsize": 12,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 6.2))
    figure.subplots_adjust(left=0.265, right=0.98, top=0.93,
                           bottom=0.16, hspace=0.63, wspace=0.18)

    table_rows = []
    for cohort_index, (cohort, cohort_label, color) in enumerate(COHORTS):
        for metric_index, (metric, metric_label) in enumerate(METRICS):
            axis = axes[cohort_index, metric_index]
            labels = []
            for method_index, (comparison, method, _) in enumerate(COMPARISONS):
                summary, pairs = selected[comparison, cohort, metric]
                mean = float(summary["delta_mean"])
                deviation = float(summary["delta_sd"])
                differences = [float(row[f"delta_{metric}"]) for row in pairs]
                labels.append(f"{method} (n={int(summary['n']):,})")
                axis.errorbar(mean, method_index, xerr=deviation,
                              fmt="none", ecolor="#303030", elinewidth=1.6,
                              capsize=4, capthick=1.6, zorder=2)
                axis.scatter(differences,
                             [method_index + offset for offset in (-0.10, 0, 0.10)],
                             s=48, color=color, alpha=0.82,
                             edgecolors="white", linewidths=0.6, zorder=3)
                axis.scatter([mean], [method_index], marker="D", s=60,
                             color="#171717", edgecolors="white",
                             linewidths=0.6, zorder=4)
                table_rows.append({
                    "method": method, "cohort": cohort_label,
                    "n": int(summary["n"]), "metric": metric_label,
                    "standardized_mean": summary["reference_mean"],
                    "reconstructed_mean": summary["variant_mean"],
                    "paired_delta_mean": summary["delta_mean"],
                    "paired_delta_sd": summary["delta_sd"],
                    "paired_training_seeds": 3,
                })
            panel_letter = "ABCD"[cohort_index * 2 + metric_index]
            axis.set_title(f"{panel_letter}  {cohort_label}",
                           loc="left", pad=12, weight="bold")
            axis.set_yticks(range(len(COMPARISONS)), labels)
            axis.tick_params(axis='y', labelleft=metric_index == 0)
            axis.set_ylim(len(COMPARISONS) - 0.55, -0.45)
            axis.set_xlabel('Change in r' if metric == 'pearson_r' else 'Change in R²', labelpad=6)
            axis.axvline(0, color="#777777", linewidth=1, linestyle=(0, (3, 3)), zorder=1)
            axis.grid(axis="x", color="#E6E6E6", linewidth=0.6)
            axis.set_axisbelow(True)
            axis.tick_params(axis="y", length=0, pad=8)
            axis.tick_params(axis="x", color="#888888")
            for spine in ("top", "right", "left"):
                axis.spines[spine].set_visible(False)
            axis.spines["bottom"].set_color("#AAAAAA")
            if metric == "pearson_r":
                axis.set_xlim(-0.28, 0.16)
                axis.set_xticks([-0.2, -0.1, 0, 0.1])
            elif cohort == "test":
                axis.set_xlim(-0.20, 0.13)
                axis.set_xticks([-0.2, -0.1, 0, 0.1])
            else:
                axis.set_xlim(-0.20, 0.70)
                axis.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
            lower, upper = axis.get_xlim()
            for comparison, _, _ in COMPARISONS:
                summary, pairs = selected[comparison, cohort, metric]
                plotted_values = [float(row[f"delta_{metric}"]) for row in pairs]
                mean, deviation = float(summary["delta_mean"]), float(summary["delta_sd"])
                plotted_values.extend((mean - deviation, mean + deviation))
                assert all(lower < value < upper for value in plotted_values), (
                    f"Clipped data or error bar: {comparison}, {cohort}, {metric}")

    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#777777",
               markeredgecolor="white", markersize=7, label="Paired seeds"),
        Line2D([], [], marker="D", color="#171717", markersize=6,
               linewidth=1.6, label="Mean ± SD"),
    ]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.59, 0.015),
                  ncol=2, frameon=False, columnspacing=2.0, handlelength=2.1, fontsize=11.5)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(args.output_dir / f"input_extent_sensitivity.{extension}", dpi=300)
    plt.close(figure)

    table_path = args.output_dir / "input_extent_summary.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    table_lines = [
        "| Method | Evaluation cohort | n | Metric | Standardized | Reconstructed | Paired change, mean ± SD |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in table_rows:
        table_lines.append(
            f"| {row['method']} | {row['cohort']} | {row['n']:,} | {row['metric']} | "
            f"{float(row['standardized_mean']):.3f} | {float(row['reconstructed_mean']):.3f} | "
            f"{float(row['paired_delta_mean']):+.3f} ± {float(row['paired_delta_sd']):.3f} |"
        )
    (args.output_dir / "input_extent_summary.md").write_text("\n".join(table_lines) + "\n")
    caption = (
        "Paired sensitivity to input extent. Each standardized/reconstructed pair was retrained "
        "on identical eligible training and validation records in grouped fold 0 and evaluated "
        "on identical eligible test or HeLa records. AttSiOff compares the standardized 19/57-nt "
        "representation with supported 21/59-nt inputs; siRNADiscovery and GNN4siRNA compare "
        "57-nt target contexts with archived source target sequences; ENsiRNA compares "
        "57-nt with reconstructed 61-nt target contexts, with structures and features "
        "recomputed for each extent. Some archived targets "
        "are fragments, so reconstructed extent does not establish recovery of every complete "
        "historical assay transcript. Circles show reconstructed-minus-standardized metric "
        "differences for training seeds 0, 1 and 2; diamonds and horizontal bars show their "
        "mean and sample standard deviation. Positive differences favor reconstructed input. "
        "Panels A–B show the matched grouped test cohorts and panels C–D the eligible HeLa "
        "subsets; exact counts appear next to each method. HeLa coverage is 43/1,047 for "
        "AttSiOff, 911/1,047 for siRNADiscovery, 942/1,047 for GNN4siRNA and 1,038/1,047 "
        "for ENsiRNA. Despite its positive "
        "HeLa R² change, AttSiOff has negative mean R² for both inputs (−1.738 and −1.421). "
        "The R² panels use different horizontal scales. These are three-seed controls on one "
        "fold, with SDs rather than confidence intervals; differing eligible cohorts preclude "
        "using the figure to rank methods.\n"
    )
    (args.output_dir / "caption.md").write_text(caption)
    manifest = {
        "analysis": "Final sensitivity-analysis-v3; all four input-extent comparisons",
        "comparisons": [item[0] for item in COMPARISONS],
        "fold": 0, "training_seeds": [0, 1, 2],
        "delta_definition": "reconstructed minus standardized",
        "uncertainty": "sample standard deviation of three paired differences",
        "verification": "All 16 plotted mean/SD pairs reproduce their three source seed values within 1e-12; reference and variant summaries also checked.",
        "inputs": {str(path): sha256(path) for path in (summary_path, seeds_path)},
        "script_sha256": sha256(Path(__file__)),
        "outputs": {path.name: sha256(path) for path in sorted(args.output_dir.iterdir())
                    if path.is_file() and path.name != "manifest.json"},
        "matplotlib_version": matplotlib.__version__,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified and plotted 16 comparisons from 48 paired metric values: {args.output_dir}")


if __name__ == "__main__":
    main()
