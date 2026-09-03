#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np

from metrics import regression_metrics


MODEL_FILES = {
    "GNN4siRNA": "gnn4sirna",
    "siRNADiscovery": "sirnadiscovery",
    "BERT-siRNA": "sirnabert",
    "OligoFormer": "oligoformer",
    "AttSiOff": "attsioff",
    "ENsiRNA": "ensirna",
}

METRICS = ("pearson", "spearman", "r2", "mae", "mse", "rmse")
LOWER_IS_BETTER = frozenset(("mae", "mse", "rmse"))


def load_labels(path: Path) -> np.ndarray:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "efficiency" not in rows[0]:
        raise ValueError(f"Missing efficiency values in {path}")
    labels = np.asarray([float(row["efficiency"]) for row in rows], dtype=float)
    if not np.all(np.isfinite(labels)):
        raise ValueError(f"Non-finite efficiency values in {path}")
    return labels


def load_competitor_predictions(path: Path, labels: np.ndarray) -> np.ndarray:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != len(labels):
        raise ValueError(
            f"{path} contains {len(rows)} predictions for {len(labels)} labels"
        )

    predictions = []
    recorded_labels = []
    for index, row in enumerate(rows):
        expected_id = f"row_{index}"
        if row.get("id") != expected_id:
            raise ValueError(
                f"{path} row {index} has id {row.get('id')!r}, expected {expected_id!r}"
            )
        recorded_labels.append(float(row["label"]))
        predictions.append(float(row["pred_label"]))

    if not np.allclose(recorded_labels, labels, rtol=0.0, atol=1e-12):
        raise ValueError(f"Labels in {path} do not match the released split")
    predictions = np.asarray(predictions, dtype=float)
    if not np.all(np.isfinite(predictions)):
        raise ValueError(f"Non-finite predictions in {path}")
    return predictions


def load_sirbench_predictions(path: Path, labels: np.ndarray) -> np.ndarray:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(labels):
        raise ValueError(
            f"{path} contains {len(rows)} predictions for {len(labels)} labels"
        )
    predictions = np.asarray([float(row["prediction"]) for row in rows], dtype=float)
    if not np.all(np.isfinite(predictions)):
        raise ValueError(f"Non-finite predictions in {path}")
    return predictions


def oriented_difference(metric: str, reference: float, comparator: float) -> float:
    if metric in LOWER_IS_BETTER:
        return comparator - reference
    return reference - comparator


def paired_bootstrap(
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    reference_name = "siRBench"
    reference = predictions[reference_name]
    comparators = [name for name in predictions if name != reference_name]
    point_metrics = {
        name: regression_metrics(labels, values)
        for name, values in predictions.items()
    }
    bootstrap_differences = {
        (name, metric): np.empty(samples, dtype=float)
        for name in comparators
        for metric in METRICS
    }

    random_generator = np.random.default_rng(seed)
    for sample_index in range(samples):
        row_indices = random_generator.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[row_indices]
        reference_metrics = regression_metrics(
            sampled_labels, reference[row_indices]
        )

        for comparator_name in comparators:
            comparator_metrics = regression_metrics(
                sampled_labels, predictions[comparator_name][row_indices]
            )
            for metric in METRICS:
                bootstrap_differences[(comparator_name, metric)][sample_index] = (
                    oriented_difference(
                        metric,
                        reference_metrics[metric],
                        comparator_metrics[metric],
                    )
                )

    results = []
    for comparator_name in comparators:
        for metric in METRICS:
            differences = bootstrap_differences[(comparator_name, metric)]
            reference_value = point_metrics[reference_name][metric]
            comparator_value = point_metrics[comparator_name][metric]
            results.append(
                {
                    "reference": reference_name,
                    "comparator": comparator_name,
                    "metric": metric,
                    "reference_value": reference_value,
                    "comparator_value": comparator_value,
                    "difference_favouring_reference": oriented_difference(
                        metric, reference_value, comparator_value
                    ),
                    "ci_lower": float(np.quantile(differences, 0.025)),
                    "ci_upper": float(np.quantile(differences, 0.975)),
                    "bootstrap_samples": samples,
                    "seed": seed,
                }
            )
    return results


def load_split_predictions(
    results_dir: Path,
    data_path: Path,
    split: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    labels = load_labels(data_path)
    competitor_filename = "preds.csv" if split == "test" else "preds_leftout.csv"
    sirbench_filename = (
        "eval_predictions_test.csv"
        if split == "test"
        else "eval_predictions_leftout.csv"
    )

    predictions = {
        display_name: load_competitor_predictions(
            results_dir / directory_name / competitor_filename,
            labels,
        )
        for display_name, directory_name in MODEL_FILES.items()
    }
    predictions["siRBench"] = load_sirbench_predictions(
        results_dir / "sirbench-model" / sirbench_filename,
        labels,
    )
    return labels, predictions


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Paired row-bootstrap comparisons against the siRBench model"
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--test-data",
        type=Path,
        default=repository_root / "data" / "siRBench_test.csv",
    )
    parser.add_argument(
        "--leftout-data",
        type=Path,
        default=repository_root / "data" / "siRBench_leftout.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root
        / "benchmark"
        / "competitors"
        / "paired_bootstrap_results.csv",
    )
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    if args.samples <= 0:
        parser.error("--samples must be positive")

    all_results = []
    for split, data_path in (
        ("test", args.test_data),
        ("leftout", args.leftout_data),
    ):
        labels, predictions = load_split_predictions(
            args.results_dir.resolve(), data_path.resolve(), split
        )
        split_results = paired_bootstrap(
            labels,
            predictions,
            samples=args.samples,
            seed=args.seed,
        )
        for row in split_results:
            row["split"] = split
        all_results.extend(split_results)

    ordered_rows = [
        {"split": row.pop("split"), **row}
        for row in all_results
    ]
    write_results(args.output.resolve(), ordered_rows)
    print(f"Saved {len(ordered_rows)} comparisons to {args.output.resolve()}")


if __name__ == "__main__":
    main()
