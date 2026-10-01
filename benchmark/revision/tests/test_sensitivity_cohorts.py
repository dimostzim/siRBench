import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_sensitivity_cohort import paired_inputs
from recover_attsioff_overhangs import recover_guides


def test_overhangs_require_unique_experimental_sequence():
    records = pd.DataFrame({'record_id': ['a', 'b', 'c'],
                            'siRNA': ['A'*19, 'C'*19, 'G'*19]})
    eligible, audit = recover_guides(records, pd.Series(
        ['a'*19+'tt', 'A'*19+'UU', 'C'*19+'AA', 'C'*19+'GG']))
    assert eligible.record_id.tolist() == ['a']
    assert eligible.siRNA.tolist() == ['A'*19+'UU']
    assert audit.status.tolist() == ['unique_original_21nt', 'conflicting_original_21nt',
                                      'original_21nt_unavailable']


def example_frames():
    partition = pd.DataFrame({'record_id': ['b', 'a', 'c'], 'efficiency': [.2, .7, .4],
                              'siRNA': ['C'*19, 'A'*19, 'G'*19],
                              'extended_mRNA': ['C'*57, 'A'*57, 'G'*57]})
    eligible = partition.iloc[[1, 0]].copy()
    eligible['standardized_context_57'] = eligible.extended_mRNA
    eligible['extended_mRNA'] = 'T'+eligible.extended_mRNA+'T'
    return partition, eligible


def test_pairing_preserves_frozen_order_and_only_changes_extent():
    partition, eligible = example_frames()
    standardized, original = paired_inputs(partition, eligible)
    assert standardized.record_id.tolist() == ['b', 'a']
    pd.testing.assert_frame_equal(standardized.drop(columns='extended_mRNA'),
                                  original.drop(columns='extended_mRNA'))
    assert original.extended_mRNA.str[1:-1].tolist() == standardized.extended_mRNA.tolist()


@pytest.mark.parametrize('column,value', [('efficiency', .99),
                                        ('standardized_context_57', 'T'*57),
                                        ('siRNA', 'U'*19)])
def test_pairing_rejects_changed_labels_or_sequence_cores(column, value):
    partition, eligible = example_frames()
    eligible.loc[eligible.index[0], column] = value
    with pytest.raises(ValueError):
        paired_inputs(partition, eligible)
