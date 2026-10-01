"""Seal both complete TabPFN matrices after artifact and training-history audit."""
import argparse
import importlib.metadata
from io import BytesIO
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from run_benchmark import POLICY, check_training_frames
from run_validation import (ROOT, PROTOCOL_SHA, FEATURE_MANIFEST_SHA, CHECKPOINT_SHA,
                            checked_bytes, sha256, write_json)

TOOLS = {'tabpfn35_frozen', 'tabpfn35_finetuned'}
IDENTITIES = set(itertools.product(['grouped', 'random'], range(5), range(3)))
ARTIFACTS = {'test_predictions': 'test_predictions.csv',
             'hela_predictions': 'hela_predictions.csv', 'train_meta': 'train_meta.json'}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def discover_attempts(runs):
    selected, incomplete = {}, []
    for attempt in sorted(runs.glob('*/fold_*/seed_*/attempt_*')):
        if not attempt.is_dir():
            continue
        identity = (attempt.parents[2].name, int(attempt.parents[1].name.removeprefix('fold_')),
                    int(attempt.parent.name.removeprefix('seed_')))
        if identity not in IDENTITIES:
            raise ValueError(f'Unexpected attempt identity: {attempt}')
        if not (attempt/'complete.json').exists():
            incomplete.append(str(attempt))
            continue
        if identity in selected:
            raise ValueError(f'Multiple completed attempts for {identity}; resolve explicitly')
        selected[identity] = attempt
    if set(selected) != IDENTITIES:
        raise ValueError(f'Incomplete matrix: {len(selected)}/30 completed identities; missing {sorted(IDENTITIES-set(selected))}')
    return selected, incomplete


def validate_prediction_frame(frame, expected):
    frame = frame.rename(columns={'id': 'record_id'})
    if frame.record_id.isna().any() or not frame.record_id.is_unique:
        raise ValueError('Invalid prediction record IDs')
    if set(frame.record_id) != set(expected.record_id):
        raise ValueError('Prediction membership differs from frozen cohort')
    if not np.isfinite(frame[['label', 'pred_label']].to_numpy(float)).all():
        raise ValueError('Nonfinite prediction or label')
    labels = expected.set_index('record_id').loc[frame.record_id, 'efficiency'].to_numpy(float)
    if not np.allclose(frame.label.to_numpy(float), labels, rtol=0, atol=1e-12):
        raise ValueError('Prediction labels differ from frozen cohort')
    return frame


def validate_reload_delta(value, predictions):
    bound = 1e-6 + 1e-5 * np.max(np.abs(predictions))
    if not np.isfinite(value) or not 0 <= value <= bound:
        raise ValueError('Recorded model-reload difference exceeds tolerance')


def validate_history(directory, metadata, initial_mse, selected_mse, variance):
    steps = read_jsonl(directory/'steps.jsonl')
    epochs = read_jsonl(directory/'epochs.jsonl')
    validation = read_jsonl(directory/'validation_audit.jsonl')
    count = len(epochs)
    if not 1 <= count <= POLICY['epochs'] or len(steps) != count or len(validation) != count+1:
        raise ValueError('Incomplete training/validation histories')
    if metadata['epochs_completed'] != count or metadata['optimizer_steps'] != count:
        raise ValueError('Training counts differ from history')
    if [row['step'] for row in steps] != list(range(1, count+1)) or [row['step'] for row in epochs] != list(range(1, count+1)):
        raise ValueError('Noncontiguous optimizer/epoch steps')
    if [row['train/epoch'] for row in steps] != list(range(count)) or [row['train/epoch'] for row in epochs] != list(range(count)):
        raise ValueError('Noncontiguous training epochs')
    if [row['train/global_step'] for row in steps] != list(range(1, count+1)):
        raise ValueError('Global optimizer steps differ from history')
    if [row['epoch'] for row in validation] != list(range(count+1)):
        raise ValueError('Noncontiguous validation checkpoints')
    losses = np.array([[row['train/loss'], row['train/lr']] for row in steps])
    values = np.array([[row['mse'], row['sample_max_parameter_change']] for row in validation])
    if not np.isfinite(losses).all() or (losses < 0).any() or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid loss, learning rate, validation error or parameter delta')
    if values[0, 1] != 0 or not (values[1:, 1] > 0).any():
        raise ValueError('No verified parameter updates from initial weights')
    for epoch, audit in zip(epochs, validation[1:]):
        if not np.isclose(epoch['val/MSE'], audit['mse'], rtol=0, atol=1e-12):
            raise ValueError('Epoch and checkpoint validation MSE disagree')
        if not np.isclose(epoch['val/r2'], 1-audit['mse']/variance, rtol=0, atol=1e-12):
            raise ValueError('Logged continuous validation R2 disagrees')
    best = min(validation, key=lambda row: row['mse'])
    if metadata['selected_epoch'] != best['epoch']:
        raise ValueError('Selected checkpoint is not the earliest global validation minimum')
    for observed, expected in [(initial_mse, validation[0]['mse']),
                               (selected_mse, best['mse']), (metadata['selected_mse'], best['mse'])]:
        if not np.isclose(observed, expected, rtol=1e-5, atol=1e-7):
            raise ValueError('Saved prediction MSE differs from selected validation checkpoint')
    if not np.isclose(metadata['selected_parameter_sample_change'], best['sample_max_parameter_change'], rtol=0, atol=1e-12):
        raise ValueError('Restored parameter delta differs from selected checkpoint')
    if count < POLICY['epochs'] and count-best['epoch'] != POLICY['patience']:
        raise ValueError('Early stopping does not match locked patience')
    return {'epochs_completed': count, 'optimizer_steps': count,
            'selected_epoch': best['epoch'], 'selected_initial_weights': best['epoch'] == 0,
            'initial_validation_r2': 1-initial_mse/variance,
            'selected_validation_r2': 1-selected_mse/variance,
            'validation_r2_gain': (initial_mse-selected_mse)/variance,
            'largest_parameter_sample_change': float(values[:, 1].max())}


def validate_native_config(config, seed):
    expected = {key: POLICY[key] for key in ['epochs', 'learning_rate', 'weight_decay', 'min_delta',
                'n_estimators_finetune', 'n_estimators_validation', 'n_estimators_final_inference',
                'use_fixed_preprocessing_seed', 'n_finetune_ctx_plus_query_samples',
                'finetune_ctx_query_split_ratio', 'grad_clip_value']}
    expected.update({'random_state': seed, 'early_stopping_patience': POLICY['patience'],
                     'early_stopping': True, 'validation_frequency': 1, 'validation_split_ratio': None,
                     'n_inference_subsample_samples': None, 'save_checkpoint_interval': None,
                     'model_version': 'v3.5', 'eval_metric': 'mse', 'use_lr_scheduler': True,
                     'lr_warmup_only': False, 'time_limit': None, 'crps_loss_weight': 1.,
                     'mse_loss_weight': 1., 'ce_loss_weight': 0., 'crls_loss_weight': 0.,
                     'mae_loss_weight': 0., 'mse_loss_clip': None})
    if any(key not in config or config[key] != value for key, value in expected.items()):
        raise ValueError('Native configuration differs from the locked training policy')


def audit_native_checkpoint(directory, metadata):
    paths = list((directory/'native').glob('*_best.pth'))
    selected_epoch = metadata['selected_epoch']
    if selected_epoch == 0:
        if paths:
            raise ValueError('Native best checkpoint exists despite selecting initial weights')
        return {'native_checkpoint_epoch': 0, 'native_optimizer_step_values': []}, []
    if len(paths) != 1:
        raise ValueError('Expected exactly one native selected checkpoint')
    import torch
    # Locally generated trusted artifacts; mmap avoids materializing full weights.
    native = torch.load(paths[0], map_location='cpu', weights_only=False, mmap=True)
    if native['epoch'] != selected_epoch or not np.isclose(native['mse'], metadata['selected_mse'], rtol=0, atol=1e-12):
        raise ValueError('Native checkpoint epoch/MSE differs from selected history')
    steps = sorted(set(float(state['step']) for state in native['optimizer']['state'].values() if 'step' in state))
    if not steps or any(not np.isfinite(step) or step != int(step) or not 0 < step <= selected_epoch for step in steps):
        raise ValueError('Invalid native optimizer step values')
    return {'native_checkpoint_epoch': selected_epoch,
            'native_optimizer_step_values': [int(step) for step in steps]}, paths


def audit_attempt(attempt, identity, context):
    from evaluate_predictions import metrics
    from prediction_artifacts import read_artifact_bytes, validate_artifact_paths

    axis, fold, seed = identity
    complete = read_json(attempt/'complete.json')
    index_path = attempt/'run_index.csv'
    if complete['status'] != 'verified' or complete['policy_version'] != POLICY['version'] or complete['run_index_sha256'] != sha256(index_path):
        raise ValueError(f'Completion/index seal mismatch: {attempt}')
    index = pd.read_csv(index_path)
    expected = {(tool, axis, fold, seed) for tool in TOOLS}
    key = ['tool', 'axis', 'fold', 'training_seed']
    if len(index) != 2 or set(map(tuple, index[key].to_numpy())) != expected or not index.status.eq('verified').all():
        raise ValueError('Attempt must contain both verified variants of one identity')
    validate_artifact_paths(index)
    lock = read_json(attempt/'policy_lock.json')
    if (lock['axis'], lock['fold'], lock['seed']) != identity or lock['policy'] != POLICY:
        raise ValueError('Attempt identity/policy differs from lock')
    for key, value in context['provenance'].items():
        if lock.get(key) != value:
            raise ValueError(f'Inconsistent provenance: {key}')
    prefix = f'{axis}/fold_{fold}'
    train, validation, test = [context['frames'][prefix+'/'+part+'.csv'] for part in ['train', 'val', 'test']]
    check_training_frames(train, validation, axis)
    expected_inputs = {part: context['manifest']['outputs'][prefix+'/'+part+'.csv'] for part in ['train', 'val']}
    if lock['input_sha256'] != expected_inputs or lock['train_ids'] != train.record_id.tolist() or lock['validation_ids'] != validation.record_id.tolist():
        raise ValueError('Recorded training/validation inputs differ from frozen split')
    if lock['train_count'] != len(train) or lock['validation_count'] != len(validation):
        raise ValueError('Recorded training/validation counts differ')
    artifacts = [attempt/'complete.json', index_path, attempt/'policy_lock.json']
    metadatas, validation_metrics = {}, {}
    for row in index.itertuples(index=False):
        directory = attempt/row.tool
        if Path(row.run_dir).resolve() != directory.resolve():
            raise ValueError('Index run directory differs from canonical returned attempt')
        for column, filename in ARTIFACTS.items():
            if Path(getattr(row, column)).resolve() != (directory/filename).resolve():
                raise ValueError('Artifact path is outside the canonical returned attempt')
            payload, _ = read_artifact_bytes(row, column)
            artifacts.append(directory/filename)
            if column != 'train_meta':
                validate_prediction_frame(pd.read_csv(BytesIO(payload)), test if column == 'test_predictions' else context['frames']['hela_full.csv'])
        metadata = read_json(directory/'train_meta.json')
        if metadata['tool'] != row.tool or metadata['status'] != 'complete' or any(metadata.get(key) != value for key, value in lock.items()):
            raise ValueError('Run metadata disagrees with policy lock')
        model_path = directory/'model.tabpfn_fit'
        if sha256(model_path) != metadata['fitted_model_sha256']:
            raise ValueError('Fitted-state hash mismatch')
        artifacts.extend([model_path, directory/'validation_predictions.csv'])
        prediction = validate_prediction_frame(pd.read_csv(directory/'validation_predictions.csv'), validation)
        computed = metrics(prediction.label.to_numpy(), prediction.pred_label.to_numpy())
        if set(metadata['validation_metrics']) != set(computed) or any(not np.isclose(metadata['validation_metrics'][key], value, rtol=0, atol=1e-12) for key, value in computed.items()):
            raise ValueError('Recorded validation metrics differ from saved predictions')
        validate_reload_delta(metadata['reload_max_abs_difference'], prediction.pred_label.to_numpy())
        if row.tool == 'tabpfn35_finetuned':
            if sha256(directory/'selected_weights.pth') != metadata['weights_sha256']:
                raise ValueError('Selected weight hash mismatch')
            validate_reload_delta(metadata['checkpoint_reload_max_abs_difference'], prediction.pred_label.to_numpy())
            artifacts.extend(directory/name for name in ['selected_weights.pth', 'steps.jsonl', 'epochs.jsonl', 'validation_audit.jsonl', 'native_configuration.json'])
        metadatas[row.tool], validation_metrics[row.tool] = metadata, computed
    tuned_dir = attempt/'tabpfn35_finetuned'
    validate_native_config(read_json(tuned_dir/'native_configuration.json'), seed)
    summary = validate_history(tuned_dir, metadatas['tabpfn35_finetuned'],
                               validation_metrics['tabpfn35_frozen']['mse'],
                               validation_metrics['tabpfn35_finetuned']['mse'],
                               float(np.var(validation.efficiency.to_numpy(float))))
    native_summary, native_paths = audit_native_checkpoint(tuned_dir, metadatas['tabpfn35_finetuned'])
    summary.update(native_summary)
    artifacts.extend(native_paths)
    summary.update({'axis': axis, 'fold': fold, 'seed': seed, 'attempt': str(attempt),
                    'train_count': len(train), 'validation_count': len(validation),
                    'elapsed_seconds': complete['elapsed_seconds'], 'host': lock['host']})
    return index, summary, {str(path): sha256(path) for path in artifacts}


def load_context(root):
    protocol = root/'evaluation/protocol-v1'
    manifest = json.loads(checked_bytes(protocol/'manifest.json', PROTOCOL_SHA))
    feature_path = root/'datasets/corrected-v1/records_features.manifest.json'
    feature_manifest = json.loads(checked_bytes(feature_path, FEATURE_MANIFEST_SHA))
    records_path = root/'datasets/corrected-v1/records_features.csv'
    records = pd.read_csv(BytesIO(checked_bytes(records_path, feature_manifest['sha256'])))
    frames = {name: pd.read_csv(BytesIO(checked_bytes(protocol/name, expected)))
              for name, expected in manifest['outputs'].items() if name.endswith('.csv')}
    revision = root/'siRBench/benchmark/revision'
    sys.path.insert(0, str(revision))
    from baselines import sequence_features
    from run_validation import representation
    representation(records, feature_manifest['features'], sequence_features)
    codes = [Path(__file__).with_name('run_benchmark.py'), Path(__file__).with_name('run_validation.py'), revision/'baselines.py']
    provenance = {'protocol_manifest_sha256': PROTOCOL_SHA, 'feature_manifest_sha256': FEATURE_MANIFEST_SHA,
                  'foundation_sha256': CHECKPOINT_SHA['v3.5'],
                  'source_commit': '9393b12a46bfc32369a89a53f6c578d22e41faba',
                  'code_sha256': {path.name: sha256(path) for path in codes},
                  'package_versions': {name: importlib.metadata.version(name) for name in ['tabpfn','torch','numpy','pandas','scikit-learn','scipy']}}
    checkpoint = root/'evaluation/tabpfn-v1/models/v3.5/tabpfn-v3.5-20260909.safetensors'
    if sha256(checkpoint) != CHECKPOINT_SHA['v3.5']:
        raise ValueError('Foundation checkpoint hash mismatch')
    return {'manifest': manifest, 'frames': frames, 'provenance': provenance, 'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.output = args.output or args.root/'evaluation/tabpfn-benchmark-v1'
    if any((args.output/name).exists() for name in ['run_index.csv', 'training_summary.csv', 'manifest.json']):
        raise ValueError('Audit/index output already exists; preserve the previous version')
    selected, incomplete = discover_attempts(args.runs.resolve())
    context = load_context(args.root)
    from evaluate_predictions import read_run_indexes, validate_predictions
    indexes, summaries, hashes = [], [], {}
    for identity, attempt in sorted(selected.items()):
        index, summary, attempt_hashes = audit_attempt(attempt, identity, context)
        indexes.append(index)
        summaries.append(summary)
        hashes.update(attempt_hashes)
    predictions = read_run_indexes([attempt/'run_index.csv' for attempt in selected.values()])
    validate_predictions(predictions, context['records'], context['frames']['membership.csv'])
    # Detect changes during collection before publishing a success marker/index.
    if any(sha256(path) != expected for path, expected in hashes.items()):
        raise ValueError('An audited artifact changed during collection')
    args.output.mkdir(parents=True, exist_ok=True)
    pd.concat(indexes, ignore_index=True).sort_values(['tool','axis','fold','training_seed']).to_csv(args.output/'run_index.csv', index=False)
    pd.DataFrame(summaries).sort_values(['axis','fold','seed']).to_csv(args.output/'training_summary.csv', index=False)
    report = {'status': 'verified', 'runs': 60, 'identities': 30, 'tools': sorted(TOOLS),
              'policy': POLICY, 'provenance': context['provenance'],
              'incomplete_attempts_preserved': incomplete,
              'index_sha256': sha256(args.output/'run_index.csv'),
              'training_summary_sha256': sha256(args.output/'training_summary.csv'),
              'artifact_sha256': hashes, 'audit_code_sha256': sha256(Path(__file__)),
              'reload_scope': 'Audits recorded per-run weight/fitted-state reload differences and saved artifact hashes; does not independently rerun GPU inference.',
              'training_step_scope': 'Logged global steps count training-step attempts; AMP can skip optimizer updates. Native best checkpoints independently record optimizer-state step values through the selected epoch, not through later discarded epochs.',
              'evaluation_scope': 'Checks held-out membership, labels and finite predictions; computes validation metrics only. No held-out performance selection or bootstrap evaluation.'}
    write_json(args.output/'manifest.json', report)
    print('Verified all 30 identities / 60 variant artifacts; canonical index sealed.')


if __name__ == '__main__':
    main()
