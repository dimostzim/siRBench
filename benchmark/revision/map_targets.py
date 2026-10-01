"""Recover target candidates from pinned source FASTAs and archived GenBank responses.

Retains every matching accession and position. A 19-nt-only match is explicitly
weaker evidence than a full context match; neither is silently made unique.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from audit_data import normalize, write_csv

FASTAS = ["upstream/gnn4sirna/data/raw/dataset_3/mRNA_3.fas",
          "repository_history/data/data_sources/Shabalina/mRNA_shabalina.fasta",
          "repository_history/data/data_sources/Simone/dataset/mrna_Simone.fas"]
XMLS = ["transcript_metadata.xml", "additional_transcripts.xml"]


def read_fasta(path):
    accession, sequence = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if accession is not None:
                yield accession, normalize("".join(sequence))
            accession, sequence = line[1:].split()[0], []
        elif line.strip():
            sequence.append(line.strip())
    if accession is not None:
        yield accession, normalize("".join(sequence))


def read_genbank(path):
    for record in ET.parse(path).findall(".//GBSeq"):
        qualifiers = defaultdict(set)
        for item in record.findall(".//GBQualifier"):
            qualifiers[item.findtext("GBQualifier_name")].add(item.findtext("GBQualifier_value"))
        yield {"accession": record.findtext("GBSeq_accession-version"),
               "accession_base": record.findtext("GBSeq_primary-accession"),
               "sequence": normalize(record.findtext("GBSeq_sequence")),
               "gene_ids": ";".join(sorted(value for value in qualifiers["db_xref"] if value.startswith("GeneID:"))),
               "gene_symbols": ";".join(sorted(qualifiers["gene"])),
               "organism": record.findtext("GBSeq_organism"),
               "molecule_type": record.findtext("GBSeq_moltype"),
               "origin": str(path), "annotation_origin": str(path)}


def references(source_root):
    refs, annotations = [], {}
    for name in XMLS:
        for record in read_genbank(source_root / name):
            refs.append(record)
            annotations[record["accession_base"]] = record
    for name in FASTAS:
        for accession, sequence in read_fasta(source_root / name):
            base = accession.split(".")[0]
            annotation = annotations.get(base, {})
            refs.append({"accession": accession, "accession_base": base, "sequence": sequence,
                         "gene_ids": annotation.get("gene_ids", ""),
                         "gene_symbols": annotation.get("gene_symbols", ""),
                         "organism": annotation.get("organism", "synthetic reporter" if base == "EGFP" else ""),
                         "molecule_type": annotation.get("molecule_type", ""),
                         "origin": str(source_root / name),
                         "annotation_origin": annotation.get("annotation_origin", "")})
    unique = {}
    for ref in refs:
        sequence_hash = hashlib.sha256(ref["sequence"].encode()).hexdigest()
        ref["sequence_sha256"] = sequence_hash
        ref["reference_id"] = ref["accession"] + ":" + sequence_hash[:16]
        if ref["reference_id"] in unique:
            unique[ref["reference_id"]]["origin"] += ";" + ref["origin"]
        else:
            unique[ref["reference_id"]] = ref
    return list(unique.values())


def find_positions(query, reference):
    if set(query) - set("ACGTNX"):
        raise ValueError("Unsupported sequence alphabet")
    pattern = "".join("[ACGTNX]" if base in "NX" else base for base in query)
    return [match.start() for match in re.finditer("(?=" + pattern + ")", reference)]


def match_record(row, refs):
    context = normalize(row["extended_mRNA"])
    query = context.strip("X")
    leading_padding = len(context) - len(context.lstrip("X"))
    matches = []
    for ref in refs:
        for position in find_positions(query, ref["sequence"]):
            target_start = position + 19 - leading_padding
            # Even if context contains unknown bases, require the entire known target.
            if ref["sequence"][target_start:target_start + 19] != normalize(row["mRNA"]):
                continue
            matches.append((ref, target_start, "context_unknown_bases" if set(query) & set("NX") else "context_exact"))
    if not matches:
        for ref in refs:
            for position in find_positions(normalize(row["mRNA"]), ref["sequence"]):
                matches.append((ref, position, "target_19nt_only"))
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.records.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    refs = references(args.source_root)
    matches, annotations = [], []
    for row in rows:
        candidates = match_record(row, refs)
        accessions, genes, qualities = set(), set(), set()
        for ref, position, quality in candidates:
            accessions.add(ref["accession_base"])
            genes.update(filter(None, ref["gene_ids"].split(";")))
            qualities.add(quality)
            matches.append({"record_id": row["record_id"], "reference_id": ref["reference_id"],
                            "accession": ref["accession"], "target_start_0based": position,
                            "target_end_0based_exclusive": position + 19, "match_quality": quality})
        annotations.append({"record_id": row["record_id"], "source": row["source"],
                            "legacy_split": row["legacy_split"], "accession_candidates": ";".join(sorted(accessions)),
                            "gene_id_candidates": ";".join(sorted(genes)),
                            "match_quality": ";".join(sorted(qualities)) or "unmapped",
                            "n_accession_candidates": len(accessions)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "target_candidates.csv", matches,
              ["record_id", "reference_id", "accession", "target_start_0based", "target_end_0based_exclusive", "match_quality"])
    write_csv(args.output_dir / "record_targets.csv", annotations)
    write_csv(args.output_dir / "references.csv", [{key: value for key, value in ref.items() if key != "sequence"} for ref in refs])
    with (args.output_dir / "references.fasta").open("w") as handle:
        for ref in refs:
            handle.write(f'>{ref["reference_id"]}\n{ref["sequence"]}\n')
    summary = {"rows": len(rows), "reference_sequences": len(refs),
               "match_quality_counts": dict(Counter(row["match_quality"] for row in annotations)),
               "ambiguous_accession_rows": sum(row["n_accession_candidates"] > 1 for row in annotations),
               "limitations": ["Candidate mappings do not identify the exact transcript used in an assay.",
                               "GenBank annotations were retrieved on 2026-09-21; archived sequences and current accession versions are distinguished.",
                               "19-nt-only mappings require further curation before original-extent reruns.",
                               "DNA references can support target identity but are not automatically valid full-length mRNA inputs."],
               "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in [args.records, *(args.source_root / name for name in FASTAS + XMLS)]]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
