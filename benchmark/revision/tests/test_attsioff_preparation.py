"""Regression checks for split-safe RNA-FM assets."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[2] / 'competitors/tools/attsioff/prepare.py'
spec = importlib.util.spec_from_file_location('attsioff_prepare', PATH)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_preparing_two_splits_keeps_their_embeddings_separate(tmp_path, monkeypatch):
    calls = []
    def fake_extract(root, fasta, output):
        calls.append(Path(fasta).name)
        lines = Path(fasta).read_text().splitlines()
        representations = Path(output) / 'representations'
        representations.mkdir(exist_ok=True)
        for identifier, sequence in zip(lines[0::2], lines[1::2]):
            np.save(representations / f'{identifier[1:]}.npy', np.array([ord(sequence[0])]))
    monkeypatch.setattr(prepare, 'run_rnafm', fake_extract)
    output = tmp_path / 'prepared'
    for name, guide in [('train', 'A'*19), ('val', 'C'*19)]:
        source = tmp_path / f'{name}.csv'
        pd.DataFrame({'record_id': [name], 'siRNA': [guide], 'extended_mRNA': ['U'*57], 'efficiency': [.4]}).to_csv(source, index=False)
        monkeypatch.setattr(sys, 'argv', ['prepare', '--input-csv', str(source), '--output-dir', str(output), '--rnafm-root', str(tmp_path), '--id-col', 'record_id'])
        prepare.main()
    train = pd.read_csv(output/'train.csv').iloc[0]
    val = pd.read_csv(output/'val.csv').iloc[0]
    assert train.RNAFM_ind != val.RNAFM_ind
    assert np.load(output/'data/RNAFM_sirna'/f'{train.RNAFM_ind}.npy').item() == ord('A')
    assert np.load(output/'data/RNAFM_sirna'/f'{val.RNAFM_ind}.npy').item() == ord('C')
    assert len(calls) == 4
    prepare.main()
    assert len(calls) == 4


def test_missing_embedding_does_not_pass_partial_cache(tmp_path, monkeypatch):
    np.save(tmp_path/'first.npy', np.zeros((19,640)))
    monkeypatch.setattr(prepare, 'run_rnafm', lambda *args: None)
    with pytest.raises(FileNotFoundError, match='second'):
        prepare.extract_missing(str(tmp_path), [('first', 'A'*19), ('second', 'C'*19)], str(tmp_path), str(tmp_path/'missing.fa'))
