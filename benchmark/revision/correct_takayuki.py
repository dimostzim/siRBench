"""Build revision base records using OligoFormer's documented strand correction.

Input is audit_data.py's canonical records.csv. Feature columns are deliberately
omitted: they describe the old sequences and must be regenerated separately.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from audit_data import normalize, pair_key, record_id, write_csv

FIELDS = ["record_id", "legacy_record_id", "legacy_split", "released_file",
          "released_line", "hela_aligned", "siRNA", "mRNA", "extended_mRNA",
          "efficiency", "source", "cell_line", "sequence_correction"]


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def correction_map(deprecated, corrected, transcript):
    if len(deprecated) != len(corrected):
        raise ValueError("Old and corrected upstream tables have different lengths")
    replacements = {}
    complement = str.maketrans("ACGTX", "TGCAX")
    for line, (old, new) in enumerate(zip(deprecated, corrected), 2):
        old_context, new_context = normalize(old["mRNA"]), normalize(new["mRNA"])
        old_guide = normalize(old["siRNA"])[::-1]
        new_guide = normalize(new["siRNA"])
        if (old_context.translate(complement) != new_context
                or old_guide.translate(complement) != new_guide
                or not math.isclose(float(old["label"]), float(new["label"]),
                                    rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f"Upstream correction identity check failed at line {line}")
        if len(new_context) != 57 or len(new_guide) != 19:
            raise ValueError(f"Unexpected sequence length at line {line}")
        if new_guide.translate(complement)[::-1] != new_context[19:38]:
            raise ValueError(f"Corrected guide does not pair with target at line {line}")
        if new_context.strip("X") not in normalize(transcript):
            raise ValueError(f"Corrected context does not match EGFP at line {line}")
        key = old_guide, old_context
        if key in replacements:
            raise ValueError(f"Duplicate upstream pair at line {line}")
        replacements[key] = {"siRNA": new_guide.replace("T", "U"),
                             "mRNA": new_context[19:38], "extended_mRNA": new_context,
                             "efficiency": float(new["label"]), "upstream_line": line}
    return replacements


def correct_records(records, replacements):
    revised, changes, seen = [], [], set()
    for old in records:
        row = {field: old[field] for field in FIELDS if field in old}
        row.update(legacy_record_id=old["record_id"], sequence_correction="none")
        key = pair_key(old)
        if old["source"].lower() == "takayuki":
            if key not in replacements or key in seen:
                raise ValueError("Takayuki record lacks a unique upstream correction")
            replacement = replacements[key]
            if float(old["efficiency"]) != round(replacement["efficiency"], 2):
                raise ValueError("Released efficacy differs from rounded upstream efficacy")
            seen.add(key)
            for field in ["siRNA", "mRNA", "extended_mRNA"]:
                row[field] = replacement[field]
            row["record_id"] = record_id(row)
            row["sequence_correction"] = "oligoformer_takayuki_strand_2025-10-31"
            changes.append({"legacy_record_id": old["record_id"], "record_id": row["record_id"],
                            "upstream_line": replacement["upstream_line"],
                            "old_siRNA": old["siRNA"], "siRNA": row["siRNA"],
                            "old_context": old["extended_mRNA"], "context": row["extended_mRNA"],
                            "efficiency": old["efficiency"]})
        revised.append(row)
    if seen != set(replacements):
        raise ValueError("Not all upstream corrections were matched to released records")
    if len({pair_key(row) for row in revised}) != len(revised):
        raise ValueError("Correction creates duplicate pairs; resolve before splitting")
    return revised, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True, help="Pinned OligoFormer checkout")
    parser.add_argument("--egfp-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    old_path = args.upstream / "data/deprecated_Taka.csv"
    new_path = args.upstream / "data/Taka.csv"
    transcript = "".join(line.strip() for line in args.egfp_fasta.read_text().splitlines()
                         if not line.startswith(">"))
    replacements = correction_map(read_csv(old_path), read_csv(new_path), transcript)
    revised, changes = correct_records(read_csv(args.records), replacements)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "records_base.csv", revised, FIELDS)
    write_csv(args.output_dir / "takayuki_corrections.csv", changes)
    manifest = {"version": "revision-corrected-v1", "rows": len(revised),
                "corrected_rows": len(changes), "features": "not yet regenerated",
                "label_policy": "Preserve released efficacy exactly; no renormalization",
                "split_policy": "Legacy membership retained as metadata, not revised evaluation splits",
                "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                           for path in [args.records, old_path, new_path, args.egfp_fasta]]}
    (args.output_dir / "correction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ["version", "rows", "corrected_rows", "features"]}))


if __name__ == "__main__":
    main()
