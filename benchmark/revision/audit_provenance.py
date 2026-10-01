"""Link recovered harmonized source rows to the released benchmark without guessing curation rules."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from audit_data import normalize, pair_key, write_csv

SOURCES = {"Huesken": "Huesken_final.csv", "Mixset": "Mixset_final.csv",
           "Shabalina": "Shabalina_final.csv", "Simone": "Simone_final.csv",
           "Takayuki": "Takayuki_final.csv"}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def audit(records, source_dir):
    released = {pair_key(row): row for row in records}
    recovered = []
    for collection, filename in SOURCES.items():
        for line, row in enumerate(read_csv(source_dir / filename), 2):
            recovered.append((collection, filename, line, row))
    pair_counts = Counter((normalize(row["siRNA"]), normalize(row["mRNA"]))
                          for _, _, _, row in recovered)
    output = []
    for collection, filename, line, row in recovered:
        key = normalize(row["siRNA"]), normalize(row["mRNA"])
        target = released.get(key)
        output.append({"collection": collection, "source_file": filename, "source_line": line,
                       "source_siRNA": row["siRNA"], "source_context": row["mRNA"],
                       "source_efficacy": row["efficacy"], "candidate_duplicate_count": pair_counts[key],
                       "released_record_id": target["record_id"] if target else "",
                       "released_source": target["source"] if target else "",
                       "released_cell_line": target["cell_line"] if target else "",
                       "released_efficacy": target["efficiency"] if target else "",
                       "matches_rounded_source_efficacy": round(float(row["efficacy"]), 2) == float(target["efficiency"]) if target else "",
                       "status": "present_in_release" if target else "absent_from_release",
                       "historical_selection_or_exclusion_rule": "not recovered; do not infer from retained value"})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = audit(read_csv(args.records), args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "source_row_audit.csv", rows)
    summary = {"recovered_source_rows": len(rows),
               "matched_released_records": len({row["released_record_id"] for row in rows if row["released_record_id"]}),
               "absent_source_rows": sum(row["status"] == "absent_from_release" for row in rows),
               "matched_rows_with_label_difference": sum(row["matches_rounded_source_efficacy"] is False for row in rows),
               "limitation": "Recovered Final tables are already transformed; this audit does not establish raw experimental provenance or a historical conflict-resolution policy.",
               "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in [args.records, *(args.source_dir / file for file in SOURCES.values())]]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
