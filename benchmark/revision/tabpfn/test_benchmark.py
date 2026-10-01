"""CPU-only guards and history checks; no model weights or held-out records."""
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import run_benchmark as runner


def records(prefix):
    return pd.DataFrame({'record_id': [prefix+'a', prefix+'b'],
                         'efficiency': [.2, .8], 'cell_line': ['H1299', 'H1299'],
                         'target_group_id': [prefix+'t1', prefix+'t2'],
                         'split_group_id': [prefix+'g1', prefix+'g2']})


def test_grouped_guard_rejects_record_target_and_component_overlap():
    train, validation = records('train'), records('validation')
    runner.check_training_frames(train, validation, 'grouped')
    for column in ['record_id', 'target_group_id', 'split_group_id']:
        invalid = validation.copy()
        invalid.loc[0, column] = train.loc[0, column]
        with pytest.raises(ValueError, match=column):
            runner.check_training_frames(train, invalid, 'grouped')


def test_random_guard_allows_target_overlap_but_never_record_overlap():
    train, validation = records('train'), records('validation')
    validation[['target_group_id', 'split_group_id']] = train[['target_group_id', 'split_group_id']]
    runner.check_training_frames(train, validation, 'random')
    validation.loc[0, 'record_id'] = train.loc[0, 'record_id']
    with pytest.raises(ValueError, match='Overlapping'):
        runner.check_training_frames(train, validation, 'random')


@pytest.mark.parametrize('axis', ['grouped', 'random'])
@pytest.mark.parametrize('cell_line', ['HeLa', ' HELA ', None])
def test_training_guard_rejects_hela_and_missing_context(axis, cell_line):
    train = records('train')
    train.loc[0, 'cell_line'] = cell_line
    with pytest.raises(ValueError, match='[Hh]eLa|cell line'):
        runner.check_training_frames(train, records('validation'), axis)


@pytest.mark.parametrize('axis', ['grouped', 'random'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -.01, 1.01])
def test_training_guard_rejects_nonfinite_and_invalid_labels(axis, value):
    validation = records('validation')
    validation.loc[0, 'efficiency'] = value
    with pytest.raises(ValueError, match='labels'):
        runner.check_training_frames(records('train'), validation, axis)


@pytest.mark.parametrize('axis', ['grouped', 'random'])
@pytest.mark.parametrize('failure', ['constant', 'empty', 'duplicate_id', 'null_id'])
def test_training_guard_rejects_unusable_frames(axis, failure):
    validation = records('validation')
    if failure == 'constant':
        validation.efficiency = .5
    elif failure == 'empty':
        validation = validation.iloc[:0]
    elif failure == 'duplicate_id':
        validation.loc[1, 'record_id'] = validation.loc[0, 'record_id']
    else:
        validation.loc[0, 'record_id'] = None
    with pytest.raises(ValueError):
        runner.check_training_frames(records('train'), validation, axis)


def test_loading_rejects_changed_bytes_even_if_labels_remain_valid(tmp_path):
    path = tmp_path/'train.csv'
    records('train').to_csv(path, index=False)
    manifest = {'outputs': {'train.csv': hashlib.sha256(path.read_bytes()).hexdigest()}}
    loaded = runner.load_frame(tmp_path, manifest, 'train.csv')
    pd.testing.assert_frame_equal(loaded, records('train'))
    records('train').assign(efficiency=[.3, .7]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='SHA256'):
        runner.load_frame(tmp_path, manifest, 'train.csv')


def test_history_retains_each_step_epoch_and_continuous_r2(tmp_path):
    logger = runner.HistoryLogger(tmp_path, variance=.1)
    logger.setup({'seed': 2, 'epochs': 100})
    for step, mse in [(1, .08), (2, .06)]:
        logger.log_step({'train/loss': .2/step, 'train/global_step': step}, step)
        logger.log_epoch({'val/MSE': mse, 'train/epoch': step-1}, step)
    steps = [json.loads(line) for line in (tmp_path/'steps.jsonl').read_text().splitlines()]
    epochs = [json.loads(line) for line in (tmp_path/'epochs.jsonl').read_text().splitlines()]
    assert steps == logger.steps and [step['step'] for step in steps] == [1, 2]
    assert epochs == logger.epochs and [epoch['train/epoch'] for epoch in epochs] == [0, 1]
    np.testing.assert_allclose([epoch['val/r2'] for epoch in epochs], [.2, .4])
    assert json.loads((tmp_path/'native_configuration.json').read_text())['seed'] == 2


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_history_fails_on_nonfinite_training_or_validation(tmp_path, value):
    logger = runner.HistoryLogger(tmp_path, variance=.1)
    with pytest.raises(ValueError, match='Nonfinite'):
        logger.log_step({'train/loss': value}, 1)
    with pytest.raises(ValueError):
        logger.log_epoch({'val/MSE': value}, 1)
    assert not (tmp_path/'steps.jsonl').exists()
    assert (tmp_path/'epochs.jsonl').read_text() == ''


def test_logger_consumes_installed_native_metric_name(tmp_path):
    from tabpfn.finetuning import FinetunedTabPFNRegressor

    native = FinetunedTabPFNRegressor(use_fixed_preprocessing_seed=False)
    logger = runner.HistoryLogger(tmp_path, variance=.1)
    metric_payload = {f'val/{native._metric_name}': .04, 'train/epoch': 0}
    logger.log_epoch(metric_payload, step=1)
    saved = json.loads((tmp_path/'epochs.jsonl').read_text())
    assert saved['val/r2'] == pytest.approx(.6)
    assert saved[f'val/{native._metric_name}'] == .04


def make_audited_regressor(monkeypatch, tmp_path, scores, parameter_samples):
    score_iterator = iter(scores)
    sample_iterator = iter(parameter_samples)

    class NativeStub:
        def _evaluate_model(self, *args, **kwargs):
            return SimpleNamespace(primary=next(score_iterator))

    monkeypatch.setitem(sys.modules, 'tabpfn.finetuning',
                        SimpleNamespace(FinetunedTabPFNRegressor=NativeStub))
    monkeypatch.setattr(runner, 'parameter_sample', lambda estimator: next(sample_iterator))
    model = runner.audited_finetuner()()
    model.finetuned_estimator_ = object()
    model.experiment_logger = runner.HistoryLogger(tmp_path, .1)
    model.experiment_logger.frozen_mse = .08
    return model


def test_audited_validation_records_initial_and_updated_weights(monkeypatch, tmp_path):
    model = make_audited_regressor(monkeypatch, tmp_path, [.08, .06],
                                  [np.array([1., 2.]), np.array([1.2, 2.])])
    model._evaluate_model()
    model._evaluate_model()
    audit = [json.loads(line) for line in (tmp_path/'validation_audit.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in audit] == [0, 1]
    assert audit[0]['sample_max_parameter_change'] == 0
    assert audit[1]['sample_max_parameter_change'] == pytest.approx(.2)
    np.testing.assert_array_equal(model.initial_parameter_sample_, [1., 2.])


def test_initial_validation_must_match_frozen_control(monkeypatch, tmp_path):
    model = make_audited_regressor(monkeypatch, tmp_path, [.09], [np.array([1., 2.])])
    with pytest.raises(ValueError, match='differs from frozen control'):
        model._evaluate_model()
    assert not (tmp_path/'validation_audit.jsonl').exists()


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_audited_validation_stops_instead_of_hiding_native_failure(monkeypatch, tmp_path, value):
    model = make_audited_regressor(monkeypatch, tmp_path, [value], [])
    with pytest.raises(ValueError, match='Native validation failed'):
        model._evaluate_model()
    assert not (tmp_path/'validation_audit.jsonl').exists()
