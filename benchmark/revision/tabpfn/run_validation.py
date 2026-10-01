"""Fixed TabPFN pilot using only frozen grouped training/validation records."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path('/SCRATCH/dtzim01/sirbench-revision-20260921')
PROTOCOL_SHA = '3a6c9eefa00843b7aa9420db3950cd55690c2a820b6636dc42916f4d3c66e9b0'
FEATURE_MANIFEST_SHA = '1affaed4e30ed26fe3de06b5215f670681fea2e419508e8506e0c18e8f21b934'
CHECKPOINT_SHA = {
    'v2': '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736',
    'v3.5': 'ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_bytes(path, expected):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f'SHA256 mismatch: {path}')
    return payload


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def validate_frames(train, validation):
    for name, frame in [('train', train), ('validation', validation)]:
        if not len(frame) or not frame.record_id.is_unique or frame.record_id.isna().any():
            raise ValueError(f'{name}: empty records or invalid/duplicate IDs')
        labels = frame.efficiency.to_numpy(float)
        if not np.isfinite(labels).all() or not ((labels >= 0) & (labels <= 1)).all():
            raise ValueError(f'{name}: labels must be finite and within [0,1]')
        if np.ptp(labels) == 0:
            raise ValueError(f'{name}: constant labels cannot support R2')
        if frame.cell_line.isna().any() or frame.cell_line.str.strip().str.lower().eq('hela').any():
            raise ValueError(f'{name}: missing cell line or HeLa record')
        if frame[['target_group_id', 'split_group_id']].isna().any().any():
            raise ValueError(f'{name}: missing group identity')
    for column in ['record_id', 'target_group_id', 'split_group_id']:
        if set(train[column]) & set(validation[column]):
            raise ValueError(f'Training/validation overlap in {column}')


def representation(frame, feature_columns, sequence_features):
    if len(feature_columns) != 100 or len(set(feature_columns)) != 100:
        raise ValueError('Expected exactly 100 unique manifest feature columns')
    prohibited = {'efficiency', 'source', 'cell_line', 'record_id', 'target_group_id',
                  'split_group_id', 'hela_aligned', 'released_line', 'legacy_split'}
    if prohibited & set(feature_columns):
        raise ValueError('Metadata or labels in feature allowlist')
    values = np.column_stack([sequence_features(frame), frame[feature_columns].to_numpy(float)])
    if values.shape != (len(frame), 176) or not np.isfinite(values).all():
        raise ValueError('Expected 176 finite feature columns')
    return values


def load_fold(protocol, manifest, fold):
    from io import BytesIO
    frames, hashes = [], {}
    for part in ['train', 'val']:
        relative = f'grouped/fold_{fold}/{part}.csv'
        expected = manifest['outputs'][relative]
        frames.append(pd.read_csv(BytesIO(checked_bytes(protocol / relative, expected))))
        hashes[relative] = expected
    validate_frames(*frames)
    return *frames, hashes


def aligned_predictions(frame, values):
    values = np.asarray(values, dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError('Predictions must be finite and aligned with validation rows')
    return pd.DataFrame({'record_id': frame.record_id, 'label': frame.efficiency,
                         'pred_label': values})


def query_checks(model, features, reference):
    reversed_predictions = np.asarray(model.predict(features[::-1]), float)[::-1]
    middle = len(features) // 2
    batched = np.concatenate([model.predict(features[:middle]), model.predict(features[middle:])])
    result = {}
    for name, values in [('reverse_order', reversed_predictions), ('two_batches', batched)]:
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite query-independence predictions: {name}')
        result[name] = {'max_absolute_difference': float(np.max(np.abs(values - reference))),
                        'allclose_rtol1e-5_atol1e-6': bool(np.allclose(values, reference, rtol=1e-5, atol=1e-6))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-version', choices=['v2', 'v3.5'], default='v2')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--folds', nargs='+', type=int, choices=range(5), default=list(range(5)))
    parser.add_argument('--seeds', nargs='+', type=int, choices=[0, 1, 2], default=[0, 1, 2])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or len(set(args.folds)) != len(args.folds) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive threads and unique folds/seeds required')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new, empty output directory')

    revision = args.root / 'siRBench/benchmark/revision'
    sys.path.insert(0, str(revision))
    from baselines import sequence_features, ridge_fit
    from evaluate_predictions import metrics
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import save_fitted_tabpfn_model
    torch.set_num_threads(args.threads)

    protocol = args.root / 'evaluation/protocol-v1'
    protocol_manifest = json.loads(checked_bytes(protocol / 'manifest.json', PROTOCOL_SHA))
    feature_manifest_path = args.root / 'datasets/corrected-v1/records_features.manifest.json'
    feature_manifest = json.loads(checked_bytes(feature_manifest_path, FEATURE_MANIFEST_SHA))
    feature_columns = feature_manifest['features']
    defaults = {'v2': 'tabpfn-v2-regressor.ckpt', 'v3.5': 'v3.5/tabpfn-v3.5-20260909.safetensors'}
    checkpoint = (args.model_path or args.root / 'evaluation/tabpfn-v1/models' / defaults[args.model_version]).resolve()
    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != CHECKPOINT_SHA[args.model_version]:
        raise ValueError('Foundation checkpoint differs from the pinned official artifact')
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        'model_version': args.model_version, 'checkpoint_path': str(checkpoint),
        'checkpoint_sha256': checkpoint_sha,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_manifest_sha256': PROTOCOL_SHA, 'feature_manifest_sha256': FEATURE_MANIFEST_SHA,
        'feature_columns': [f'guide_pos{position}_{base}' for position in range(1,20) for base in 'ACGU'] + feature_columns,
        'policy': 'Validation-only; frozen train rows only; released TabPFN preprocessing; 8 estimators; no gradient fine-tuning, clipping, calibration or train+validation refit.',
        'seed_meaning': 'TabPFN preprocessing/ensemble random_state, not a gradient-training replicate.',
        'folds': args.folds, 'seeds': args.seeds, 'threads': args.threads,
        'package_versions': {name: importlib.metadata.version(name) for name in ['tabpfn','torch','numpy','pandas','scikit-learn','scipy']},
        'source_commit': '9393b12a46bfc32369a89a53f6c578d22e41faba',
        'code_sha256': {str(path): sha256(path) for path in [Path(__file__).resolve(), revision/'baselines.py', revision/'evaluate_predictions.py']},
    }
    write_json(args.output / 'manifest.json', provenance)
    all_metrics, ridge_metrics = [], []
    for fold in args.folds:
        train, validation, input_hashes = load_fold(protocol, protocol_manifest, fold)
        train_x = representation(train, feature_columns, sequence_features)
        validation_x = representation(validation, feature_columns, sequence_features)
        train_y = train.efficiency.to_numpy(float)
        validation_y = validation.efficiency.to_numpy(float)
        fold_dir = args.output / f'fold_{fold}'
        fold_dir.mkdir()
        write_json(fold_dir/'inputs.json', {'sha256': input_hashes, 'train_count':len(train),
                   'validation_count':len(validation), 'train_ids':train.record_id.tolist(),
                   'validation_ids':validation.record_id.tolist()})
        started = time.perf_counter()
        ridge, selection = ridge_fit(train_x, validation_x, train_y, validation_y)
        ridge_prediction = aligned_predictions(validation, ridge.predict(validation_x))
        ridge_prediction.to_csv(fold_dir/'ridge_validation_predictions.csv', index=False)
        ridge_result = {'model_version':'guide_thermodynamic_ridge', 'fold':fold, 'seed':-1,
                        'n_train':len(train), 'n_validation':len(validation),
                        'fit_seconds':time.perf_counter()-started, **metrics(validation_y, ridge_prediction.pred_label)}
        write_json(fold_dir/'ridge_selection.json', {**selection, 'metrics':ridge_result,
                   'interpretation':'Alpha selected on these validation labels; validation performance is selection-optimistic, not held-out test evidence.'})
        ridge_metrics.append(ridge_result)
        pd.DataFrame(ridge_metrics).to_csv(args.output/'ridge_metrics.csv', index=False)
        for seed in args.seeds:
            run_dir = fold_dir / f'seed_{seed}'
            run_dir.mkdir()
            version = ModelVersion.V2 if args.model_version == 'v2' else ModelVersion.V3_5
            model = TabPFNRegressor.create_default_for_version(
                version, model_path=str(checkpoint), n_estimators=8,
                random_state=seed, device=args.device)
            started = time.perf_counter()
            model.fit(train_x, train_y)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            values = model.predict(validation_x)
            predict_seconds = time.perf_counter() - started
            predictions = aligned_predictions(validation, values)
            prediction_path = run_dir/'validation_predictions.csv'
            predictions.to_csv(prediction_path, index=False)
            fitted_path = run_dir/'model.tabpfn_fit'
            save_fitted_tabpfn_model(model, fitted_path)
            checks = query_checks(model, validation_x, values) if seed == args.seeds[0] else None
            result = {'model_version':args.model_version, 'fold':fold, 'seed':seed,
                      'n_train':len(train), 'n_validation':len(validation),
                      'fit_seconds':fit_seconds, 'predict_seconds':predict_seconds,
                      **metrics(validation_y, values)}
            write_json(run_dir/'run.json', {**result, 'status':'complete',
                       'train_count':len(train), 'validation_count':len(validation),
                       'constructor_params':model.get_params(deep=False),
                       'resolved_n_estimators':int(model.n_estimators_),
                       'query_checks':checks, 'checkpoint_sha256':checkpoint_sha,
                       'predictions_sha256':sha256(prediction_path), 'fitted_model_sha256':sha256(fitted_path)})
            all_metrics.append(result)
            pd.DataFrame(all_metrics).to_csv(args.output/'metrics.csv', index=False)
            print(json.dumps(result), flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_json(args.output/'complete.json', {'runs':len(args.folds)*len(args.seeds),
               'metrics_sha256':sha256(args.output/'metrics.csv')})


if __name__ == '__main__':
    main()
