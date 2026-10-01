import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_splits import check_partition, partitions, sequence_groups, target_groups


def test_ambiguous_accessions_and_gene_aliases_stay_together():
    annotations = [
        {"record_id": "a", "accession_candidates": "NM_A;NM_B", "gene_id_candidates": "GeneID:1"},
        {"record_id": "b", "accession_candidates": "NM_B", "gene_id_candidates": "GeneID:1"},
        {"record_id": "c", "accession_candidates": "NM_C", "gene_id_candidates": "GeneID:1"},
        {"record_id": "d", "accession_candidates": "NM_D", "gene_id_candidates": "GeneID:2"},
    ]
    groups = target_groups(annotations)
    assert groups['a'] == groups['b'] == groups['c']
    assert groups['a'] != groups['d']
    assert groups == target_groups(list(reversed(annotations)))


def test_near_guides_link_nonhela_groups_without_using_hela_sequences():
    rows = [{"record_id": key, "legacy_split": split, "siRNA": seq} for key, split, seq in [
        ('a', 'train', 'A' * 19), ('b', 'test', 'A' * 16 + 'CCC'),
        ('c', 'hela_full', 'A' * 15 + 'CCCC')]]
    groups, links = sequence_groups(rows, {'a': 'group_a', 'b': 'group_b', 'c': 'group_c'})
    assert groups['a'] == groups['b'] and groups['b'] != groups['c']
    assert links == [{'first_id': 'a', 'second_id': 'b', 'hamming_distance': 3}]


def fixture():
    rows = [{"record_id": str(index), "legacy_split": "train", "efficiency": str(index / 60)} for index in range(60)]
    rows.append({"record_id": "hela", "legacy_split": "hela_full", "efficiency": "0.5"})
    groups = {str(index): 'target_' + str(index // 3) for index in range(60)}
    groups['hela'] = 'reporter'
    return rows, groups


def test_partitions_cover_nonhela_once_and_never_overlap_groups():
    rows, groups = fixture()
    heldout = Counter()
    for axis, fold, parts in partitions(rows, groups):
        check_partition(rows, groups, parts, axis == 'grouped')
        if axis == 'grouped':
            heldout.update(parts['test'])
    assert heldout == Counter({index: 1 for index in range(60)})


def test_membership_does_not_depend_on_labels():
    rows, groups = fixture()
    before = [(axis, fold, {key: list(value) for key, value in parts.items()}) for axis, fold, parts in partitions(rows, groups)]
    for index, row in enumerate(rows):
        row['efficiency'] = str((index % 2))
    after = [(axis, fold, {key: list(value) for key, value in parts.items()}) for axis, fold, parts in partitions(rows, groups)]
    assert before == after


def test_group_overlap_is_rejected():
    rows = [{'record_id': 'a', 'legacy_split': 'train'}, {'record_id': 'b', 'legacy_split': 'test'}, {'record_id': 'c', 'legacy_split': 'validation'}]
    with pytest.raises(ValueError, match='leakage'):
        check_partition(rows, {'a':'same', 'b':'same', 'c':'other'}, {'train':[0], 'test':[1], 'val':[2]}, True)
