"""Prepare dataset records and label-blind splits, then verify the published hashes.

The starting point is the released harmonized data, not a reconstruction of
unavailable historical curation decisions. Features may be reused or regenerated.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

from download import digest, verify_files

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from audit_data import audit, normalize, write_csv
from correct_takayuki import FIELDS, correct_records, read_csv


def audited_replacements(changes):
    replacements = {}
    complement = str.maketrans("ACGTX", "TGCAX")
    for row in changes:
        guide, context = normalize(row["siRNA"]), normalize(row["context"])
        old_guide, old_context = normalize(row["old_siRNA"]), normalize(row["old_context"])
        if (old_guide.translate(complement) != guide
                or old_context.translate(complement) != context
                or guide.translate(complement)[::-1] != context[19:38]):
            raise ValueError("Correction does not satisfy the audited strand transformation")
        key = old_guide, old_context
        if key in replacements:
            raise ValueError("Duplicate correction")
        replacements[key] = {"siRNA": row["siRNA"], "mRNA": context[19:38],
                             "extended_mRNA": row["context"],
                             "efficiency": float(row["efficiency"]),
                             "upstream_line": row["upstream_line"]}
    return replacements


def require_identical(actual, expected):
    if digest(actual) != digest(expected):
        raise ValueError(f"Rebuilt file differs from the archive: {actual}")


def export_reference_inputs(protocol, output):
    for fold in range(5):
        for part, name in [("train", "train"), ("val", "validation"), ("test", "test")]:
            rows = read_csv(protocol / "grouped" / f"fold_{fold}" / f"{part}.csv")
            split = output / f"fold_{fold}" / name
            (split / "input").mkdir(parents=True)
            write_csv(split / "input/data.csv", [
                {"id": row["record_id"], "siRNA": row["siRNA"],
                 "extended_mRNA": row["extended_mRNA"]} for row in rows])
            write_csv(split / "labels.csv", [
                {"id": row["record_id"], "label": row["efficiency"]} for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", choices=["archived", "regenerate"], default="archived")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    package, output = args.package.resolve(), args.output.resolve()
    verify_files(package)
    output.mkdir(parents=True, exist_ok=False)
    work = package / "workspace"
    frozen = work / "datasets/corrected-v1"
    audit(work / "siRBench/data", output / "audit")
    require_identical(output / "audit/records.csv", work / "audit/data/records.csv")
    revised, changes = correct_records(
        read_csv(output / "audit/records.csv"),
        audited_replacements(read_csv(frozen / "takayuki_corrections.csv")))
    dataset = output / "datasets/corrected-v1"
    dataset.mkdir(parents=True)
    write_csv(dataset / "records_base.csv", revised, FIELDS)
    write_csv(dataset / "takayuki_corrections.csv", changes)
    for name in ["records_base.csv", "takayuki_corrections.csv"]:
        require_identical(dataset / name, frozen / name)
    features = dataset / "records_features.csv"
    if args.features == "regenerate":
        for tool in ["RNAfold", "RNAcofold", "RNAup"]:
            version = subprocess.check_output([tool, "--version"], text=True).strip()
            if version != f"{tool} 2.4.18":
                raise ValueError(f"Benchmark requires {tool} 2.4.18, found {version}")
        subprocess.run([sys.executable, str(SCRIPT_DIR / "regenerate_features.py"),
                        "--input", str(dataset / "records_base.csv"), "--output", str(features),
                        "--workers", str(args.workers)], check=True)
    else:
        shutil.copy2(frozen / "records_features.csv", features)
    require_identical(features, frozen / "records_features.csv")
    # Preserve the sealed feature manifest expected by TabPFN's input guards.
    shutil.copy2(frozen / "records_features.manifest.json", dataset / "records_features.manifest.json")
    protocol = output / "evaluation/protocol-v1"
    subprocess.run([sys.executable, str(SCRIPT_DIR / "make_splits.py"),
                    "--records", str(features), "--targets", str(work / "audit/targets/record_targets.csv"),
                    "--output-dir", str(protocol)], check=True)
    expected = json.loads((work / "evaluation/protocol-v1/manifest.json").read_text())
    verified = []
    for name, checksum in expected["outputs"].items():
        if name == "run_matrix.csv":
            continue  # Absolute input paths are intentionally rooted at the new output.
        if digest(protocol / name) != checksum:
            raise ValueError(f"Rebuilt protocol differs: {name}")
        verified.append(name)
    with (protocol / "run_matrix.csv").open() as handle:
        actual = list(csv.DictReader(handle))
    with (work / "evaluation/protocol-v1/run_matrix.csv").open() as handle:
        original = list(csv.DictReader(handle))
    for row in actual:
        for key in ["train", "val", "test", "hela_full"]:
            row[key] = str(Path(row[key]).relative_to(protocol))
    for row in original:
        for key in ["train", "val", "test", "hela_full"]:
            row[key] = row[key].split("evaluation/protocol-v1/", 1)[1]
    if actual != original:
        raise ValueError("Rebuilt run matrix differs after normalizing paths")
    export_reference_inputs(protocol, output / "reference-training")
    report = {"status": "PASS", "rows": len(revised), "corrected_rows": len(changes),
              "features": args.features, "verified_protocol_files": verified,
              "run_matrix": "identical after path normalization",
              "starting_point": "released harmonized CSVs and archived correction/target audits"}
    (output / "dataset-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
