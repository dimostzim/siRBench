"""Audit the released benchmark without changing records, labels, or splits."""
import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

FILES = {
    "development": "siRBench_train.csv",
    "train": "val_split/siRBench_train_split.csv",
    "validation": "val_split/siRBench_val_split.csv",
    "test": "siRBench_test.csv",
    "hela_full": "leftout/siRBench_hela.csv",
    "hela_aligned": "siRBench_leftout.csv",
    "hela_removed": "leftout/siRBench_leftout_dropped.csv",
}


def normalize(sequence):
    return sequence.upper().replace("U", "T")


def pair_key(row):
    return normalize(row["siRNA"]), normalize(row["extended_mRNA"])


def record_id(row):
    # Label-independent IDs survive repartitioning and efficacy corrections.
    return "sb_" + hashlib.sha256("|".join(pair_key(row)).encode()).hexdigest()[:20]


def read_records(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [pair_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate sequence-context pairs in {path}")
    for row in rows:
        efficacy = float(row["efficiency"])
        if not np.isfinite(efficacy) or not 0 <= efficacy <= 1:
            raise ValueError(f"Invalid efficacy in {path}: {efficacy}")
    return rows


def record_signature(row):
    return pair_key(row) + (row["source"], row["cell_line"], float(row["efficiency"]))


def partition_matches(whole, *parts):
    return Counter(map(record_signature, whole)) == sum(
        (Counter(map(record_signature, part)) for part in parts), Counter()
    )


def close_sequence_pairs(rows, max_mismatches=3):
    sequences = [normalize(row["siRNA"]) for row in rows]
    if len({len(seq) for seq in sequences}) != 1:
        raise ValueError("Hamming audit requires equal-length siRNAs")
    encoded = np.array([list(seq.encode()) for seq in sequences], dtype=np.uint8)
    for offset in range(0, len(rows), 128):
        distance = np.count_nonzero(encoded[offset:offset + 128, None] != encoded, axis=2)
        for local, other in np.argwhere(distance <= max_mismatches):
            first = offset + int(local)
            if first < other:
                yield first, int(other), int(distance[local, other])


def write_csv(path, rows, fields=None):
    rows = list(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(data_root, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {name: read_records(data_root / path) for name, path in FILES.items()}
    checks = {
        "development_equals_train_plus_validation": partition_matches(
            datasets["development"], datasets["train"], datasets["validation"]),
        "full_hela_equals_aligned_plus_removed": partition_matches(
            datasets["hela_full"], datasets["hela_aligned"], datasets["hela_removed"]),
    }
    inputs = []
    for name, rows in datasets.items():
        path = data_root / FILES[name]
        inputs.append({"dataset": name, "path": FILES[name], "rows": len(rows),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_csv(output_dir / "input_manifest.csv", inputs)

    canonical = []
    aligned = {pair_key(row) for row in datasets["hela_aligned"]}
    composition = []
    primary_names = ["train", "validation", "test", "hela_full"]
    for name in primary_names:
        for line, row in enumerate(datasets[name], 2):
            canonical.append({"record_id": record_id(row), "legacy_split": name,
                              "released_file": FILES[name], "released_line": line,
                              "hela_aligned": pair_key(row) in aligned, **row})
        for (source, cell), count in Counter(
                (row["source"], row["cell_line"]) for row in datasets[name]).items():
            composition.append({"split": name, "source": source, "cell_line": cell, "n": count})
    checks["canonical_ids_unique"] = len({row["record_id"] for row in canonical}) == len(canonical)
    complement = str.maketrans("ACGT", "TGCA")
    checks["target_equals_guide_reverse_complement"] = all(
        normalize(row["siRNA"]).translate(complement)[::-1] == normalize(row["mRNA"])
        for row in canonical)
    checks["target_is_central_19mer"] = all(
        normalize(row["extended_mRNA"])[19:38] == normalize(row["mRNA"])
        for row in canonical)
    feature_columns = [key for key in canonical[0] if key not in {
        "record_id", "legacy_split", "released_file", "released_line", "hela_aligned",
        "siRNA", "mRNA", "extended_mRNA", "efficiency", "source", "cell_line"}]
    checks["features_finite"] = all(
        np.isfinite(float(row[column])) for row in canonical for column in feature_columns)
    write_csv(output_dir / "records.csv", canonical)
    write_csv(output_dir / "composition.csv", composition)

    overlaps = []
    for first, second in itertools.combinations(primary_names, 2):
        for field in ["siRNA", "mRNA", "extended_mRNA"]:
            left = {normalize(row[field]) for row in datasets[first]}
            right = {normalize(row[field]) for row in datasets[second]}
            overlaps.append({"first_split": first, "second_split": second,
                             "field": field, "shared_sequences": len(left & right)})
    write_csv(output_dir / "exact_overlap.csv", overlaps)
    close_pairs = []
    for first, second, distance in close_sequence_pairs(canonical):
        left, right = canonical[first], canonical[second]
        close_pairs.append({"first_id": left["record_id"], "second_id": right["record_id"],
                            "first_split": left["legacy_split"], "second_split": right["legacy_split"],
                            "hamming_distance": distance})
    write_csv(output_dir / "similar_sequence_pairs.csv", close_pairs,
              ["first_id", "second_id", "first_split", "second_split", "hamming_distance"])
    graph_degrees = {field: max(Counter(normalize(row[field]) for row in canonical).values())
                     for field in ["siRNA", "extended_mRNA"]}
    summary = {
        "checks": checks, "rows_full": len(canonical),
        "rows_aligned": sum(len(datasets[name]) for name in ["train", "validation", "test", "hela_aligned"]),
        "removed_hela_rows": len(datasets["hela_removed"]), "numeric_features": len(feature_columns),
        "rows_with_context_padding": sum("X" in row["extended_mRNA"] for row in canonical),
        "maximum_interactions_per_graph_sequence_node": graph_degrees,
        "cross_split_hamming_pairs": {str(threshold): sum(
            pair["hamming_distance"] <= threshold and pair["first_split"] != pair["second_split"]
            for pair in close_pairs) for threshold in [0, 1, 2, 3]},
        "limitations": ["No gene/transcript grouping is inferred from a 57-nt context.",
                        "Sequence proximity thresholds are descriptive audits, not selected split rules.",
                        "Released metadata are recorded, not independently verified source provenance."],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(args.data_root, args.output_dir)
    print(json.dumps(summary, indent=2))
    if not all(summary["checks"].values()):
        raise SystemExit("Dataset audit has failed checks; inspect summary.json")


if __name__ == "__main__":
    main()
