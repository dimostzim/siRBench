import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).resolve().parents[2] / 'competitors/tools/ensirna/subset_features.py'
sys.path.insert(0, str(MODULE.parent))
spec = importlib.util.spec_from_file_location('ensirna_subset', MODULE)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_reordered_subset_injects_labels_by_id_without_mutating_canonical_features():
    records = {name: {'id': name, 'anti seq': 'ACGU', 'sense seq': 'ACGU', 'mRNA_seq': 'ACGU',
                      'efficiency': value} for name, value in [('one', 0.1), ('two', 0.2)]}
    features = {name: [SimpleNamespace(benchmark_record_id=name), value, 'feature']
                for name, value in [('one', 0.1), ('two', 0.2)]}
    rows = [{'record_id': name, 'siRNA': 'ACGT', 'extended_mRNA': 'XACGTX', 'efficiency': value}
            for name, value in [('two', 0.7), ('one', 0.9)]]
    selected, feature_rows = wrapper.select_records(rows, records, features)
    assert [(row['id'], row['efficiency']) for row in selected] == [('two', 0.7), ('one', 0.9)]
    assert [(row[0].benchmark_record_id, row[1]) for row in feature_rows] == [('two', 0.7), ('one', 0.9)]
    assert features['one'][1] == 0.1


def test_matching_id_cannot_reuse_features_for_changed_sequence():
    records = {'one': {'anti seq': 'ACGU', 'sense seq': 'ACGU', 'mRNA_seq': 'ACGU'}}
    rows = [{'record_id': 'one', 'siRNA': 'AAAA', 'extended_mRNA': 'ACGU', 'efficiency': 0.1}]
    with pytest.raises(ValueError, match='Sequence mismatch'):
        wrapper.select_records(rows, records, {})


def test_canonical_part_order_is_mapped_by_embedded_record_id(tmp_path):
    import json
    import pickle

    records = tmp_path / 'all.jsonl'
    records.write_text('{"id":"one"}\n{"id":"two"}\n')
    part = tmp_path / 'part_0.pkl'
    features = [[SimpleNamespace(benchmark_record_id='two'), 0.2],
                [SimpleNamespace(benchmark_record_id='one'), 0.1]]
    with part.open('wb') as handle:
        pickle.dump(features, handle)
    (tmp_path / '_metainfo').write_text(json.dumps({'num_entry': 2, 'file_names': [str(part)],
                                                   'file_num_entries': [2]}))
    _, loaded, _ = wrapper.load_canonical(records, tmp_path)
    assert (loaded['one'][1], loaded['two'][1]) == (0.1, 0.2)
