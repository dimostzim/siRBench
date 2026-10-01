"""Freeze label-blind target/sequence-grouped folds and random-split sensitivity runs."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ShuffleSplit

from audit_data import close_sequence_pairs, write_csv

SPLIT_SEED = 20260921
TRAINING_SEEDS = [0, 1, 2]


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def join(self, keys):
        first, *others = keys
        for key in others:
            self.parent[self.find(key)] = self.find(first)
        return self.find(first)

    def stable_ids(self):
        members = defaultdict(list)
        for key in list(self.parent):
            members[self.find(key)].append(key)
        return {root: "group_" + hashlib.sha256("|".join(sorted(keys)).encode()).hexdigest()[:16]
                for root, keys in members.items()}


def target_groups(annotations):
    components = Components()
    record_roots = {}
    for row in sorted(annotations, key=lambda row: row["record_id"]):
        keys = ["accession:" + key for key in row["accession_candidates"].split(";") if key]
        keys += [key for key in row["gene_id_candidates"].split(";") if key]
        if not keys or row["record_id"] in record_roots:
            raise ValueError("Every record needs a unique annotated target mapping")
        record_roots[row["record_id"]] = components.join(keys)
    stable = components.stable_ids()
    return {record: stable[components.find(root)] for record, root in record_roots.items()}


def sequence_groups(rows, groups):
    components = Components()
    for group in groups.values():
        components.find(group)
    development = [row for row in rows if row["legacy_split"] != "hela_full"]
    links = []
    for first, second, distance in close_sequence_pairs(development, max_mismatches=3):
        left, right = development[first]["record_id"], development[second]["record_id"]
        components.join([groups[left], groups[right]])
        links.append({"first_id": left, "second_id": right, "hamming_distance": distance})
    stable = components.stable_ids()
    return {record: stable[components.find(group)] for record, group in groups.items()}, links


def partitions(rows, groups):
    # Only indices and unsupervised group IDs enter either splitter; no efficacy labels.
    eligible = np.array([index for index, row in enumerate(rows) if row["legacy_split"] != "hela_full"])
    group_labels = np.array([groups[rows[index]["record_id"]] for index in eligible])
    outer = GroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    for fold, (development, test) in enumerate(outer.split(eligible, groups=group_labels)):
        inner = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SPLIT_SEED + fold)
        train, validation = next(inner.split(development, groups=group_labels[development]))
        yield "grouped", fold, {"train": eligible[development[train]], "val": eligible[development[validation]], "test": eligible[test]}
    random = ShuffleSplit(n_splits=5, test_size=0.09, random_state=SPLIT_SEED)
    for fold, (development, test) in enumerate(random.split(eligible)):
        inner = ShuffleSplit(n_splits=1, test_size=0.10, random_state=SPLIT_SEED + fold)
        train, validation = next(inner.split(development))
        yield "random", fold, {"train": eligible[development[train]], "val": eligible[development[validation]], "test": eligible[test]}


def check_partition(rows, groups, parts, grouped):
    expected = {index for index, row in enumerate(rows) if row["legacy_split"] != "hela_full"}
    sets = {name: set(indices) for name, indices in parts.items()}
    if set.union(*sets.values()) != expected or sum(map(len, sets.values())) != len(expected):
        raise ValueError("Partition must cover every non-HeLa row exactly once")
    for first, second in itertools.combinations(sets, 2):
        if sets[first] & sets[second]:
            raise ValueError("Overlapping rows")
        left = {groups[rows[index]["record_id"]] for index in sets[first]}
        right = {groups[rows[index]["record_id"]] for index in sets[second]}
        if grouped and left & right:
            raise ValueError("Grouped partition has target/sequence leakage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Protocol output must be empty; frozen splits are never overwritten")
    with args.records.open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["record_id"])
    with args.targets.open(newline="") as handle:
        annotations = list(csv.DictReader(handle))
    targets = target_groups(annotations)
    if len(rows) != len(targets) or {row["record_id"] for row in rows} != set(targets):
        raise ValueError("Record and target annotation IDs differ")
    groups, sequence_links = sequence_groups(rows, targets)
    for row in rows:
        row.update(target_group_id=targets[row["record_id"]], split_group_id=groups[row["record_id"]])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "groups.csv", [{key: row[key] for key in ["record_id", "target_group_id", "split_group_id"]} for row in rows])
    write_csv(args.output_dir / "similar_sequence_links.csv", sequence_links, ["first_id", "second_id", "hamming_distance"])
    hela = [row for row in rows if row["legacy_split"] == "hela_full"]
    write_csv(args.output_dir / "hela_full.csv", hela)
    write_csv(args.output_dir / "hela_aligned.csv", [row for row in hela if row["hela_aligned"].lower() == "true"])
    membership, sizes, runs, grouped_test_counts = [], [], [], Counter()
    for axis, fold, parts in partitions(rows, groups):
        check_partition(rows, groups, parts, grouped=axis == "grouped")
        directory = args.output_dir / axis / f"fold_{fold}"
        directory.mkdir(parents=True)
        for part, indices in parts.items():
            selected = [rows[index] for index in sorted(indices)]
            write_csv(directory / (part + ".csv"), selected)
            membership.extend({"axis": axis, "fold": fold, "part": part, "record_id": row["record_id"]} for row in selected)
            sizes.append({"axis": axis, "fold": fold, "part": part, "rows": len(selected),
                          "target_groups": len({row["target_group_id"] for row in selected}),
                          "split_groups": len({row["split_group_id"] for row in selected})})
        if axis == "grouped":
            grouped_test_counts.update(rows[index]["record_id"] for index in parts["test"])
        for seed in TRAINING_SEEDS:
            runs.append({"axis": axis, "fold": fold, "training_seed": seed,
                         "train": str(directory / "train.csv"), "val": str(directory / "val.csv"),
                         "test": str(directory / "test.csv"), "hela_full": str(args.output_dir / "hela_full.csv")})
    if set(grouped_test_counts.values()) != {1}:
        raise ValueError("Every non-HeLa record must be tested in exactly one grouped fold")
    write_csv(args.output_dir / "membership.csv", membership)
    write_csv(args.output_dir / "partition_sizes.csv", sizes)
    write_csv(args.output_dir / "run_matrix.csv", runs)
    manifest = {"protocol": "benchmark-revision-v1", "primary_axis": "grouped", "split_seed": SPLIT_SEED,
                "folds_per_axis": 5, "training_seeds": TRAINING_SEEDS,
                "group_rule": "Connected candidate accessions/GeneIDs, then non-HeLa guide pairs with Hamming distance <=3/19; no outcome data",
                "validation_rule": "10% of remaining groups for grouped folds; 10% of remaining rows for random splits",
                "random_test_fraction": 0.09, "hela_policy": "All 1047 rows held out from fitting/selection; aligned subset secondary only",
                "inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.records, args.targets, Path(__file__)]},
                "outputs": {str(path.relative_to(args.output_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(args.output_dir.rglob("*.csv"))}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "hela_rows": len(hela), "training_runs_per_method": len(runs),
                      "nonhela_target_groups": len({row['target_group_id'] for row in rows if row['legacy_split'] != 'hela_full'}),
                      "nonhela_split_groups": len({row['split_group_id'] for row in rows if row['legacy_split'] != 'hela_full'})}, indent=2))


if __name__ == "__main__":
    main()
