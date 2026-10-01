"""Check the wrapper boundary into the published GNN preprocessing scripts."""
import importlib.util
from pathlib import Path
import sys

import pandas as pd

PATH = Path(__file__).resolve().parents[2] / 'competitors/tools/gnn4sirna/prepare.py'
spec = importlib.util.spec_from_file_location('gnn_prepare', PATH)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_preserves_record_id_and_rna_alphabet_for_thermodynamics(tmp_path, monkeypatch):
    source = tmp_path / 'input.csv'
    output = tmp_path / 'prepared'
    pd.DataFrame({'record_id':['sb_example'], 'siRNA':['AU'*9+'A'],
                  'extended_mRNA':['ATG'*19], 'efficiency':[.6]}).to_csv(source, index=False)
    monkeypatch.setattr(prepare, 'run_preprocess', lambda *args: None)
    monkeypatch.setattr(prepare.os.path, 'isdir', lambda path: True)
    monkeypatch.setattr(sys, 'argv', ['prepare', '--input-csv',str(source),'--output-dir',str(output)])
    prepare.main()
    prepared = pd.read_csv(output / 'input.csv')
    assert prepared.id.tolist() == ['sb_example']
    fasta = (output / 'raw/input/sirna.fas').read_text().splitlines()[1]
    assert fasta == 'AU'*9+'A'
