import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from map_targets import find_positions, match_record


def test_retains_overlapping_positions():
    assert find_positions("AAA", "AAAAA") == [0, 1, 2]


def test_padding_does_not_shift_target_coordinate():
    row = {"mRNA": "A" * 19, "extended_mRNA": "X" * 19 + "A" * 19 + "C" * 19}
    refs = [{"sequence": "A" * 19 + "C" * 19}]
    assert match_record(row, refs) == [(refs[0], 0, "context_exact")]


def test_context_unknown_is_reported_and_known_target_must_match():
    row = {"mRNA": "A" * 19, "extended_mRNA": "C" * 18 + "X" + "A" * 19 + "G" * 19}
    refs = [{"sequence": "C" * 19 + "A" * 19 + "G" * 19}]
    assert match_record(row, refs) == [(refs[0], 19, "context_unknown_bases")]


def test_target_only_is_weaker_evidence_and_preserves_ambiguity():
    row = {"mRNA": "A" * 19, "extended_mRNA": "C" * 19 + "A" * 19 + "G" * 19}
    refs = [{"sequence": "T" + "A" * 19}, {"sequence": "G" + "A" * 19}]
    assert match_record(row, refs) == [(ref, 1, "target_19nt_only") for ref in refs]
