import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

MODULE = Path(__file__).resolve().parents[2] / "competitors/tools/ensirna/prepare.py"
spec = importlib.util.spec_from_file_location("ensirna_prepare", MODULE)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_preserves_internal_unknown_base_positions():
    assert wrapper.clean_seq("XXXATGXCATNN") == "AUGXCAU"


def test_missing_sense_target_fails_instead_of_using_antisense_or_position_zero():
    with pytest.raises(ValueError, match="Target site is absent"):
        wrapper.resolve_position("AAAA", "UUUU", "AAAA")


def test_existing_pdb_keeps_record_id_without_rosetta(monkeypatch, tmp_path):
    pdb = tmp_path / "existing.pdb"
    pdb.write_text("fixture")
    source = tmp_path / "input.csv"
    output = tmp_path / "output.jsonl"
    pd.DataFrame([{"record_id": "sb_unique", "siRNA": "ACGU", "mRNA": "ACGU",
                   "extended_mRNA": "XACGUX", "efficiency": .4,
                   "pdb_data_path": str(pdb), "chain": "[0,1]", "start": "[0,1]"}]).to_csv(source, index=False)
    monkeypatch.delenv("ROSETTA_DIR", raising=False)
    monkeypatch.setattr(wrapper.sys, "argv", [str(MODULE), "--input-csv", str(source),
                                             "--output-jsonl", str(output)])
    wrapper.main()
    result = json.loads(output.read_text())
    assert (result["id"], result["position"], result["mRNA_seq"]) == ("sb_unique", 0, "ACGU")


def test_missing_later_pdb_does_not_leave_a_partial_prepared_file(monkeypatch, tmp_path):
    source = tmp_path / 'input.csv'
    output = tmp_path / 'output.jsonl'
    existing = tmp_path / 'existing.pdb'
    existing.write_text('fixture')
    rows = [{'record_id': f'sb_{number}', 'siRNA': 'ACGU', 'mRNA': 'ACGU',
             'extended_mRNA': 'ACGU', 'efficiency': 0.4, 'pdb_data_path': str(path),
             'chain': '[0,1]', 'start': '[0,1]'}
            for number, path in enumerate([existing, tmp_path / 'missing.pdb'])]
    pd.DataFrame(rows).to_csv(source, index=False)
    monkeypatch.setattr(wrapper.sys, 'argv', [str(MODULE), '--input-csv', str(source),
                                             '--output-jsonl', str(output)])
    with pytest.raises(FileNotFoundError):
        wrapper.main()
    assert not output.exists()
