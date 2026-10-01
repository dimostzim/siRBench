"""Synthetic artifact failures for the independent TabPFN benchmark auditor."""
import json

import numpy as np
import pandas as pd
import pytest

import audit_benchmark as auditor
import summarize_benchmark_training as training_summary


def write_json(path, value):
    path.write_text(json.dumps(value)+'\n')


def write_history(directory, count=22, selected=2, initial=.0625, best=.04, variance=.09):
    steps, epochs, validation = [], [], []
    for epoch in range(count+1):
        mse = initial if epoch == 0 else best if epoch == selected else initial+.01
        validation.append({'epoch': epoch, 'mse': mse, 'sample_max_parameter_change': epoch*.01})
        if epoch:
            steps.append({'step': epoch, 'train/epoch': epoch-1, 'train/global_step': epoch,
                          'train/loss': .1, 'train/lr': 1e-5})
            epochs.append({'step': epoch, 'train/epoch': epoch-1, 'val/MSE': mse, 'val/r2': 1-mse/variance})
    for name, rows in [('steps.jsonl', steps), ('epochs.jsonl', epochs), ('validation_audit.jsonl', validation)]:
        (directory/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return {'epochs_completed': count, 'optimizer_steps': count, 'selected_epoch': selected,
            'selected_mse': best, 'selected_parameter_sample_change': selected*.01}


def test_complete_discovery_preserves_failed_attempts_and_rejects_ambiguity(tmp_path):
    for axis, fold, seed in auditor.IDENTITIES:
        attempt = tmp_path/axis/f'fold_{fold}'/f'seed_{seed}'/'attempt_2'
        attempt.mkdir(parents=True)
        write_json(attempt/'complete.json', {})
    failed = tmp_path/'grouped/fold_0/seed_0/attempt_1'
    failed.mkdir()
    selected, incomplete = auditor.discover_attempts(tmp_path)
    assert len(selected) == 30 and incomplete == [str(failed)]
    write_json(failed/'complete.json', {})
    with pytest.raises(ValueError, match='Multiple completed'):
        auditor.discover_attempts(tmp_path)


def test_incomplete_matrix_never_passes(tmp_path):
    with pytest.raises(ValueError, match='0/30'):
        auditor.discover_attempts(tmp_path)


def test_prediction_ids_labels_and_finite_values_are_required():
    expected = pd.DataFrame({'record_id': ['a', 'b'], 'efficiency': [.2, .8]})
    predictions = pd.DataFrame({'record_id': ['b', 'a'], 'label': [.8, .2], 'pred_label': [.7, .3]})
    auditor.validate_prediction_frame(predictions, expected)
    for changed, message in [(predictions.iloc[:1], 'membership'),
                             (predictions.assign(record_id=['a', 'a']), 'IDs'),
                             (predictions.assign(label=[.2, .8]), 'labels'),
                             (predictions.assign(pred_label=[np.nan, .3]), 'Nonfinite')]:
        with pytest.raises(ValueError, match=message):
            auditor.validate_prediction_frame(changed, expected)


def test_history_checks_best_checkpoint_and_native_epoch_payload(tmp_path):
    metadata = write_history(tmp_path)
    summary = auditor.validate_history(tmp_path, metadata, .0625, .04, .09)
    assert summary['selected_epoch'] == 2 and summary['epochs_completed'] == 22
    with pytest.raises(ValueError, match='global validation minimum'):
        auditor.validate_history(tmp_path, {**metadata, 'selected_epoch': 3}, .0625, .04, .09)
    with pytest.raises(ValueError, match='checkpoint'):
        auditor.validate_history(tmp_path, metadata, .0625, .05, .09)
    with pytest.raises(ValueError, match='parameter delta'):
        auditor.validate_history(tmp_path, {**metadata, 'selected_parameter_sample_change': .5}, .0625, .04, .09)
    write_history(tmp_path, count=21)
    with pytest.raises(ValueError, match='patience'):
        auditor.validate_history(tmp_path, {**metadata, 'epochs_completed': 21, 'optimizer_steps': 21}, .0625, .04, .09)


def test_initial_checkpoint_can_remain_selected_after_real_updates(tmp_path):
    metadata = write_history(tmp_path, count=20, selected=0, initial=.04, best=.04)
    summary = auditor.validate_history(tmp_path, metadata, .04, .04, .09)
    assert summary['selected_initial_weights'] and summary['validation_r2_gain'] == 0


@pytest.fixture
def attempt_fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(auditor.ROOT/'siRBench/benchmark/revision'))
    from evaluate_predictions import metrics
    from tabpfn.finetuning import FinetunedTabPFNRegressor
    from tabpfn.constants import ModelVersion
    import torch

    attempt = tmp_path/'attempt_2'
    attempt.mkdir()
    frames = {}
    for part, prefix in [('train', 't'), ('val', 'v'), ('test', 'q')]:
        frames[f'grouped/fold_0/{part}.csv'] = pd.DataFrame({
            'record_id': [prefix+'1', prefix+'2'], 'efficiency': [.2, .8],
            'cell_line': ['H1299']*2, 'target_group_id': [prefix+'g1', prefix+'g2'],
            'split_group_id': [prefix+'g1', prefix+'g2']})
    frames['hela_full.csv'] = pd.DataFrame({'record_id': ['h1', 'h2'], 'efficiency': [.2, .8]})
    provenance = {'code_sha256': {'run_benchmark.py': 'expected-hash'}}
    lock = {'axis': 'grouped', 'fold': 0, 'seed': 0, 'policy': auditor.POLICY,
            **provenance, 'input_sha256': {'train': 'train-hash', 'val': 'val-hash'},
            'train_ids': ['t1', 't2'], 'validation_ids': ['v1', 'v2'],
            'train_count': 2, 'validation_count': 2, 'host': 'synthetic'}
    write_json(attempt/'policy_lock.json', lock)
    indexes = []
    for tool in sorted(auditor.TOOLS):
        directory = attempt/tool
        directory.mkdir()
        values = [.4, .6] if tool.endswith('finetuned') else [.45, .55]
        validation = pd.DataFrame({'record_id': ['v1', 'v2'], 'label': [.2, .8], 'pred_label': values})
        validation.to_csv(directory/'validation_predictions.csv', index=False)
        (directory/'model.tabpfn_fit').write_bytes(b'synthetic fitted state')
        metadata = {**lock, 'tool': tool, 'status': 'complete',
                    'fitted_model_sha256': auditor.sha256(directory/'model.tabpfn_fit'),
                    'validation_metrics': metrics(validation.label, validation.pred_label), 'reload_max_abs_difference': 0.}
        if tool.endswith('finetuned'):
            metadata.update(write_history(directory))
            (directory/'selected_weights.pth').write_bytes(b'synthetic selected weights')
            metadata.update({'weights_sha256': auditor.sha256(directory/'selected_weights.pth'),
                             'checkpoint_reload_max_abs_difference': 0.})
            config = FinetunedTabPFNRegressor(model_version=ModelVersion.V3_5, epochs=100,
                random_state=0, validation_split_ratio=None, early_stopping_patience=20,
                min_delta=0., n_estimators_finetune=2, n_estimators_validation=8,
                n_estimators_final_inference=8, use_fixed_preprocessing_seed=False,
                save_checkpoint_interval=None).get_params()
            config['model_version'] = 'v3.5'
            config['eval_metric'] = 'mse'
            write_json(directory/'native_configuration.json', config)
            (directory/'native').mkdir()
            torch.save({'epoch': 2, 'mse': .04,
                        'optimizer': {'state': {0: {'step': torch.tensor(2.)}}}},
                       directory/'native/checkpoint_best.pth')
        write_json(directory/'train_meta.json', metadata)
        for cohort, ids in [('test', ['q1', 'q2']), ('hela', ['h1', 'h2'])]:
            pd.DataFrame({'record_id': ids, 'label': [.2, .8], 'pred_label': [.4, .6]}).to_csv(directory/f'{cohort}_predictions.csv', index=False)
        row = {'tool': tool, 'axis': 'grouped', 'fold': 0, 'training_seed': 0,
               'run_dir': str(directory), 'status': 'verified'}
        for column, filename in auditor.ARTIFACTS.items():
            row[column] = str(directory/filename)
            row[column+'_sha256'] = auditor.sha256(directory/filename)
        indexes.append(row)
    pd.DataFrame(indexes).to_csv(attempt/'run_index.csv', index=False)
    write_json(attempt/'complete.json', {'status': 'verified', 'policy_version': auditor.POLICY['version'],
               'run_index_sha256': auditor.sha256(attempt/'run_index.csv'), 'elapsed_seconds': 1.})
    context = {'frames': frames, 'provenance': provenance,
               'manifest': {'outputs': {'grouped/fold_0/train.csv': 'train-hash', 'grouped/fold_0/val.csv': 'val-hash'}}}
    return attempt, context


def test_real_artifact_helpers_accept_valid_synthetic_attempt(attempt_fixture):
    attempt, context = attempt_fixture
    index, summary, hashes = auditor.audit_attempt(attempt, ('grouped', 0, 0), context)
    assert len(index) == 2 and summary['selected_epoch'] == 2
    assert str(attempt/'tabpfn35_finetuned/selected_weights.pth') in hashes
    assert summary['native_optimizer_step_values'] == [2]


@pytest.mark.parametrize('change, message', [('index', 'seal mismatch'), ('weights', 'weight hash'),
                                            ('code', 'provenance'), ('prediction', 'SHA256')])
def test_completed_artifact_tampering_rejected(attempt_fixture, change, message):
    attempt, context = attempt_fixture
    if change == 'index':
        path = attempt/'run_index.csv'
        path.write_text(path.read_text()+'\n')
    elif change == 'weights':
        (attempt/'tabpfn35_finetuned/selected_weights.pth').write_bytes(b'changed')
    elif change == 'code':
        context['provenance'] = {'code_sha256': {'run_benchmark.py': 'changed'}}
    else:
        path = attempt/'tabpfn35_frozen/test_predictions.csv'
        path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match=message):
        auditor.audit_attempt(attempt, ('grouped', 0, 0), context)


def test_native_checkpoint_metadata_and_amp_step_scope(attempt_fixture):
    import torch
    attempt, _ = attempt_fixture
    directory = attempt/'tabpfn35_finetuned'
    metadata = json.loads((directory/'train_meta.json').read_text())
    path = directory/'native/checkpoint_best.pth'
    checkpoint = {'epoch': 2, 'mse': .04, 'optimizer': {'state': {0: {'step': torch.tensor(1.)}}}}
    torch.save(checkpoint, path)
    summary, _ = auditor.audit_native_checkpoint(directory, metadata)
    assert summary['native_optimizer_step_values'] == [1]
    checkpoint['epoch'] = 3
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='epoch/MSE'):
        auditor.audit_native_checkpoint(directory, metadata)
    with pytest.raises(ValueError, match='initial weights'):
        auditor.audit_native_checkpoint(directory, {**metadata, 'selected_epoch': 0})


def test_training_plot_uses_only_validation_summary(tmp_path):
    frame = pd.DataFrame([{'axis': axis, 'fold': fold, 'seed': seed,
                          'initial_validation_r2': .2, 'selected_validation_r2': .3,
                          'validation_r2_gain': .1, 'epochs_completed': 25, 'selected_epoch': 5}
                         for axis, fold, seed in auditor.IDENTITIES])
    training_summary.summarize(frame, tmp_path)
    assert (tmp_path/'training_validation_summary.png').stat().st_size > 1000
    assert len(pd.read_csv(tmp_path/'training_summary_by_axis.csv')) == 2
    with pytest.raises(ValueError, match='30 unique'):
        training_summary.validate_summary(frame.iloc[:-1])
