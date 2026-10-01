import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "audit_data.py"
spec = importlib.util.spec_from_file_location("audit_data", MODULE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(sequence, efficacy="0.5"):
    return {"siRNA": sequence, "extended_mRNA": "A" * 57, "efficiency": efficacy,
            "source": "study", "cell_line": "cell"}


def test_record_id_does_not_depend_on_label_or_dna_rna_alphabet():
    assert audit.record_id(row("ACGU", "0.2")) == audit.record_id(row("ACGT", "0.8"))


def test_partition_matching_checks_labels_and_multiplicity():
    whole = [row("AAAA"), row("CCCC")]
    assert audit.partition_matches(whole, whole[:1], whole[1:])
    assert not audit.partition_matches(whole, whole[:1], [row("CCCC", "0.9")])
    assert not audit.partition_matches(whole, whole, whole[:1])


def test_hamming_audit_reports_each_pair_once():
    rows = [row("AAAA"), row("AAAT"), row("AATT"), row("CCCC")]
    assert list(audit.close_sequence_pairs(rows, 1)) == [(0, 1, 1), (1, 2, 1)]


def test_hamming_audit_rejects_incomparable_lengths():
    with pytest.raises(ValueError, match="equal-length"):
        list(audit.close_sequence_pairs([row("AAA"), row("AAAA")]))


def test_duplicate_pairs_are_rejected(tmp_path):
    path = tmp_path / "data.csv"
    audit.write_csv(path, [row("AAAA"), row("AAAA", "0.9")])
    with pytest.raises(ValueError, match="Duplicate"):
        audit.read_records(path)


def test_nonfinite_labels_are_rejected(tmp_path):
    path = tmp_path / "data.csv"
    audit.write_csv(path, [row("AAAA", "nan")])
    with pytest.raises(ValueError, match="Invalid efficacy"):
        audit.read_records(path)
