import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('run_validation', Path(__file__).with_name('run_validation.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def frame(prefix):
    data = pd.DataFrame({'record_id':[prefix+'a',prefix+'b'], 'efficiency':[.2,.8],
                         'cell_line':['H1299','H1299'], 'source':['x','y'],
                         'target_group_id':[prefix+'t1',prefix+'t2'],
                         'split_group_id':[prefix+'g1',prefix+'g2'],
                         'siRNA':['A'*19,'C'*19]})
    return pd.concat([data, pd.DataFrame({f'f{i}':[i,i+1] for i in range(100)})], axis=1)


def encode(data):
    return np.array([[base == token for base in seq for token in 'ACGU'] for seq in data.siRNA],float)


def test_explicit_features_ignore_changed_metadata():
    data = frame('a')
    columns = [f'f{i}' for i in range(100)]
    expected = runner.representation(data, columns, encode)
    changed = data.assign(source='other',cell_line='other',efficiency=.9,record_id=['q','z'])
    np.testing.assert_array_equal(runner.representation(changed, columns, encode),expected)
    assert expected.shape == (2,176)
    with pytest.raises(ValueError,match='Metadata'):
        runner.representation(data, columns[:-1]+['efficiency'], encode)


def test_disjoint_records_targets_components_and_no_hela():
    train, validation = frame('a'),frame('b')
    runner.validate_frames(train,validation)
    for column in ['record_id','target_group_id','split_group_id']:
        bad = validation.copy()
        bad.loc[0,column] = train.loc[0,column]
        with pytest.raises(ValueError,match=column):
            runner.validate_frames(train,bad)
    with pytest.raises(ValueError,match='HeLa'):
        runner.validate_frames(train,validation.assign(cell_line='HeLa'))


@pytest.mark.parametrize('value',[np.nan,np.inf,-.01,1.01])
def test_invalid_labels_rejected(value):
    validation = frame('b')
    validation.loc[0,'efficiency'] = value
    with pytest.raises(ValueError,match='labels'):
        runner.validate_frames(frame('a'),validation)


def test_sha_change_rejected_even_for_valid_range_label(tmp_path):
    path = tmp_path/'train.csv'
    data = frame('a')
    data.to_csv(path,index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert runner.checked_bytes(path,digest) == path.read_bytes()
    data.loc[0,'efficiency'] = .3
    data.to_csv(path,index=False)
    with pytest.raises(ValueError,match='SHA256'):
        runner.checked_bytes(path,digest)


def test_nonfinite_features_and_misaligned_predictions_rejected():
    data = frame('a')
    data.loc[0,'f5'] = np.nan
    with pytest.raises(ValueError,match='finite'):
        runner.representation(data,[f'f{i}' for i in range(100)],encode)
    for values in [[.5],[np.nan,.5]]:
        with pytest.raises(ValueError,match='aligned'):
            runner.aligned_predictions(data,values)


def test_loader_opens_only_train_and_validation(tmp_path):
    folder = tmp_path/'grouped/fold_0'
    folder.mkdir(parents=True)
    manifest = {'outputs':{}}
    for part,prefix in [('train','a'),('val','b')]:
        path = folder/f'{part}.csv'
        frame(prefix).to_csv(path,index=False)
        manifest['outputs'][str(path.relative_to(tmp_path))] = runner.sha256(path)
    train,validation,hashes = runner.load_fold(tmp_path,manifest,0)
    assert len(train) == len(validation) == 2
    assert set(hashes) == {'grouped/fold_0/train.csv','grouped/fold_0/val.csv'}
    assert not (folder/'test.csv').exists()


def test_query_checks_detect_batch_dependence():
    class Stable:
        def predict(self,values):
            return values[:,0]
    features = np.arange(8).reshape(4,2)
    checks = runner.query_checks(Stable(),features,features[:,0])
    assert all(check['allclose_rtol1e-5_atol1e-6'] for check in checks.values())
    class Dependent:
        def predict(self,values):
            return values[:,0] + len(values)
    checks = runner.query_checks(Dependent(),features,Dependent().predict(features))
    assert not checks['two_batches']['allclose_rtol1e-5_atol1e-6']
