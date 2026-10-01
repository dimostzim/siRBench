import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from correct_takayuki import correct_records, correction_map
from audit_data import record_id


def example():
    guide = "CCUCGCCCUUGCUCACCAU"
    context = "XXXXXXXXXXXXXXXXXXXATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGT"
    complement = str.maketrans("ACGTX", "TGCAX")
    old_context = context.translate(complement)
    old_guide = guide.replace("U", "T").translate(complement)
    old = {"siRNA": old_guide[::-1], "mRNA": old_context, "label": "0.5587304827023166"}
    new = {"siRNA": guide, "mRNA": context, "label": old["label"]}
    released = {"siRNA": old_guide, "mRNA": old_context[19:38], "extended_mRNA": old_context,
                "source": "Takayuki", "efficiency": "0.56", "cell_line": "HeLa",
                "legacy_split": "hela_full", "released_file": "example.csv", "released_line": "2",
                "hela_aligned": "True", "DG_total": "-999"}
    released["record_id"] = record_id(released)
    return old, new, released, context.strip("X")


def test_correction_preserves_labels_and_identity_link_but_drops_old_features():
    old, new, released, transcript = example()
    mapping = correction_map([old], [new], transcript)
    revised, changes = correct_records([released], mapping)
    assert revised[0]["siRNA"] == new["siRNA"]
    assert revised[0]["efficiency"] == released["efficiency"]
    assert revised[0]["legacy_record_id"] == released["record_id"]
    assert revised[0]["record_id"] != released["record_id"]
    assert "DG_total" not in revised[0]
    assert len(changes) == 1


def test_rejects_context_not_found_in_transcript():
    old, new, _, _ = example()
    with pytest.raises(ValueError, match="EGFP"):
        correction_map([old], [new], "A" * 100)


def test_rejects_unexpected_upstream_label_change():
    old, new, _, transcript = example()
    new["label"] = "0.7"
    with pytest.raises(ValueError, match="identity"):
        correction_map([old], [new], transcript)


def test_rejects_released_label_mismatch():
    old, new, released, transcript = example()
    released["efficiency"] = "0.6"
    with pytest.raises(ValueError, match="efficacy"):
        correct_records([released], correction_map([old], [new], transcript))


def test_rejects_duplicate_introduced_by_correction():
    old, new, released, transcript = example()
    other = {**released, "siRNA": new["siRNA"], "mRNA": new["mRNA"][19:38],
             "extended_mRNA": new["mRNA"], "source": "other"}
    other["record_id"] = record_id(other)
    with pytest.raises(ValueError, match="duplicate"):
        correct_records([released, other], correction_map([old], [new], transcript))


def test_accepts_upstream_float_serialization_roundoff():
    old, new, _, transcript = example()
    old["label"] = "0.12450250025512802"
    new["label"] = "0.124502500255128"
    assert len(correction_map([old], [new], transcript)) == 1
