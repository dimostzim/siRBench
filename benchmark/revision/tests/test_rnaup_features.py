import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "data/scripts"))
import make_all_features as features


@pytest.mark.parametrize("summary, expected", [
    ("(-38.03 = -39.40 + 0.44 + 0.93)", (1.37, -39.40, -38.03)),
    ("(-38.96 = -39.40 + 0.44)", (0.44, -39.40, -38.96)),
])
def test_rnaup_components_follow_vienna_output_order(monkeypatch, summary, expected):
    # The four-term fixture is actual ViennaRNA 2.4.18 stdout for this duplex.
    stdout = "(((((&))))) 1,19 : 1,19 " + summary + "\nSEQUENCE&SEQUENCE\n"
    monkeypatch.setattr(features.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=stdout, stderr=""))
    result = features.try_rnaup_energies("CCUCGCCCUUGCUCACCAU", "AUGGUGAGCAAGGGCGAGG")
    assert result == pytest.approx(expected)
