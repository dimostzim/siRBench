import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subset_gnn_features import subset_features


def fixture(tmp_path):
    source = tmp_path/'canonical'
    features = source/'processed/all'
    features.mkdir(parents=True)
    pd.DataFrame({'id':['a','b'],'siRNA':['s1','s2'],'mRNA':['m1','m1'],
        'efficiency':[.1,.2],'siRNA_seq':['A'*19,'T'*19],
        'mRNA_seq':['C'*100,'C'*100]}).to_csv(source/'all.csv',index=False)
    pd.DataFrame([['s1',1,2],['s2',3,4]]).to_csv(features/'sirna_kmers.txt',index=False,header=False)
    pd.DataFrame([['m1',5,6]]).to_csv(features/'target_kmers.txt',index=False,header=False)
    pd.DataFrame([['s1','m1',7,8],['s2','m1',9,10]]).to_csv(features/'sirna_target_thermo.csv',index=False,header=False)
    records = pd.DataFrame({'record_id':['b','a'],'siRNA':['U'*19,'A'*19],
                           'extended_mRNA':['C'*100,'C'*100],'efficiency':[.8,.7]})
    return source,records,tmp_path/'subset'


def test_features_follow_ids_and_current_labels_not_canonical_order(tmp_path):
    source,records,destination = fixture(tmp_path)
    subset_features(source,records,destination,'test')
    result = pd.read_csv(destination/'test.csv')
    assert result.id.tolist() == ['b','a']
    np.testing.assert_allclose(result.efficiency,[.8,.7])
    pairs = pd.read_csv(destination/'processed/test/sirna_target_thermo.csv',header=None)
    assert pairs[2].tolist() == [9,7]
    assert len(pd.read_csv(destination/'processed/test/target_kmers.txt',header=None)) == 1


@pytest.mark.parametrize('column,value',[('siRNA','C'*19),('extended_mRNA','A'*100),
                                        ('record_id','missing')])
def test_wrong_sequences_or_ids_are_rejected(tmp_path,column,value):
    source,records,destination = fixture(tmp_path)
    records.loc[0,column] = value
    with pytest.raises(ValueError):
        subset_features(source,records,destination,'test')


def test_identical_shared_target_rows_become_one_node(tmp_path):
    source,records,destination = fixture(tmp_path)
    path = source/'processed/all/target_kmers.txt'
    path.write_text('m1,5,6\nm1,5,6\n')
    subset_features(source,records,destination,'test')
    actual = pd.read_csv(destination/'processed/test/target_kmers.txt',header=None)
    assert actual.values.tolist() == [['m1',5,6]]


def test_conflicting_shared_target_features_are_rejected(tmp_path):
    source,records,destination = fixture(tmp_path)
    path = source/'processed/all/target_kmers.txt'
    path.write_text('m1,5,6\nm1,5,7\n')
    with pytest.raises(ValueError,match='Conflicting canonical features'):
        subset_features(source,records,destination,'test')
    assert not (destination/'test.csv').exists()
