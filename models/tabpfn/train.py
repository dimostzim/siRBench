"""Frozen and gradient-fine-tuned TabPFN-3.5 on one sealed benchmark split."""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
from io import BytesIO
import json
from pathlib import Path
import random
import socket
import sys
import time

import numpy as np
import pandas as pd

from common import (PROTOCOL_SHA, FEATURE_MANIFEST_SHA, CHECKPOINT_SHA,
                            checked_bytes, sha256, write_json, representation,
                            aligned_predictions, validate_frames)

INFERENCE_CONFIG = {'ENABLE_GPU_PREPROCESSING': False, 'TRANSFORM_TEXT': False,
                    'TRANSFORM_DATES': False, 'SUBSAMPLE_SAMPLES': None}
POLICY = {
    'version': 'tabpfn35-benchmark-v1', 'epochs': 100, 'patience': 20,
    'learning_rate': 1e-5, 'weight_decay': 0.01, 'min_delta': 0.0,
    'n_estimators_finetune': 2, 'n_estimators_validation': 8,
    'n_estimators_final_inference': 8, 'use_fixed_preprocessing_seed': False,
    'n_finetune_ctx_plus_query_samples': 50000, 'finetune_ctx_query_split_ratio': 0.2,
    'selection': 'Minimum complete-validation MSE, equivalent to maximum R2; initial weights eligible',
    'optimizer': 'AdamW', 'scheduler': '10% linear warmup then cosine decay',
    'loss': 'released CRPS + MSE, unit weights', 'grad_clip_value': 1.0,
    'inference_config': INFERENCE_CONFIG,
    'features': '76 guide one-hot + 100 manifest guide/central-duplex features; no assay metadata',
    'inference': 'Training rows only; eight estimators; one complete-cohort predict call',
    'restart': 'Fresh attempt from foundation weights; no native automatic resume',
}


def load_frame(protocol, manifest, relative):
    return pd.read_csv(BytesIO(checked_bytes(protocol / relative, manifest['outputs'][relative])))


def check_training_frames(train, validation, axis):
    if axis == 'grouped':
        validate_frames(train, validation)
    else:
        # Random splits deliberately allow target/sequence-group overlap.
        for frame in (train, validation):
            labels = frame.efficiency.to_numpy(float)
            if not len(frame) or not frame.record_id.is_unique or frame.record_id.isna().any():
                raise ValueError('Invalid training/validation record IDs')
            if not np.isfinite(labels).all() or np.ptp(labels) == 0 or not ((0 <= labels) & (labels <= 1)).all():
                raise ValueError('Invalid training/validation labels')
            if frame.cell_line.isna().any() or frame.cell_line.str.strip().str.lower().eq('hela').any():
                raise ValueError('HeLa or missing cell line in training/validation')
        if set(train.record_id) & set(validation.record_id):
            raise ValueError('Overlapping training/validation IDs')


class HistoryLogger:
    def __init__(self, output, variance):
        self.output = output
        self.variance = variance
        self.epochs = []
        self.steps = []

    def append(self, filename, value):
        with (self.output / filename).open('a') as handle:
            handle.write(json.dumps(value, allow_nan=False, default=str) + '\n')

    def setup(self, config):
        write_json(self.output / 'native_configuration.json', config)

    def log_step(self, metrics, step):
        if not np.isfinite(metrics['train/loss']):
            raise ValueError('Nonfinite training loss')
        value = {'step': step, **metrics}
        self.steps.append(value)
        self.append('steps.jsonl', value)

    def log_epoch(self, metrics, step):
        value = {'step': step, **metrics, 'val/r2': 1 - metrics['val/MSE'] / self.variance}
        self.epochs.append(value)
        self.append('epochs.jsonl', value)
        print(json.dumps(value, allow_nan=False), flush=True)

    def finish(self):
        pass


def parameter_sample(estimator):
    # Fixed positions across every trainable tensor demonstrate real weight updates.
    import torch
    return torch.cat([parameter.detach().flatten()[:16].float().cpu()
                      for parameter in estimator.model_.parameters() if parameter.requires_grad]).numpy()


def audited_finetuner():
    from tabpfn.finetuning import FinetunedTabPFNRegressor

    class AuditedRegressor(FinetunedTabPFNRegressor):
        def _evaluate_model(self, *args, **kwargs):
            result = super()._evaluate_model(*args, **kwargs)
            if not np.isfinite(result.primary):
                raise ValueError('Native validation failed or returned nonfinite MSE')
            sample = parameter_sample(self.finetuned_estimator_)
            if not hasattr(self, 'validation_audit_'):
                self.initial_parameter_sample_ = sample.copy()
                self.validation_audit_ = []
                if not np.isclose(result.primary, self.experiment_logger.frozen_mse, rtol=1e-5, atol=1e-7):
                    raise ValueError('Initial fine-tuning validation differs from frozen control')
            value = {'epoch': len(self.validation_audit_), 'mse': float(result.primary),
                     'sample_max_parameter_change': float(np.max(np.abs(sample - self.initial_parameter_sample_)))}
            self.validation_audit_.append(value)
            self.experiment_logger.append('validation_audit.jsonl', value)
            return result

    return AuditedRegressor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--axis', choices=['grouped', 'random'], required=True)
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--seed', type=int, choices=range(3), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    benchmark_dir = Path(__file__).resolve().parents[2] / 'benchmark'
    sys.path.insert(0, str(benchmark_dir))
    from baselines import sequence_features
    from evaluate_predictions import metrics
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import (save_tabpfn_model, save_fitted_tabpfn_model,
                                      load_fitted_tabpfn_model)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    started = time.monotonic()
    protocol = args.root / 'evaluation/protocol-v1'
    protocol_manifest = json.loads(checked_bytes(protocol / 'manifest.json', PROTOCOL_SHA))
    feature_manifest = args.root / 'datasets/corrected-v1/records_features.manifest.json'
    columns = json.loads(checked_bytes(feature_manifest, FEATURE_MANIFEST_SHA))['features']
    checkpoint = args.root / 'evaluation/tabpfn-v1/models/v3.5/tabpfn-v3.5-20260909.safetensors'
    if sha256(checkpoint) != CHECKPOINT_SHA['v3.5']:
        raise ValueError('Foundation checkpoint differs from pinned artifact')
    prefix = f'{args.axis}/fold_{args.fold}'
    train = load_frame(protocol, protocol_manifest, prefix + '/train.csv')
    validation = load_frame(protocol, protocol_manifest, prefix + '/val.csv')
    check_training_frames(train, validation, args.axis)
    train_x = representation(train, columns, sequence_features)
    val_x = representation(validation, columns, sequence_features)
    train_y = train.efficiency.to_numpy(float)
    val_y = validation.efficiency.to_numpy(float)
    common = {
        'axis': args.axis, 'fold': args.fold, 'seed': args.seed,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
        'gpu': torch.cuda.get_device_name(), 'policy': POLICY,
        'protocol_manifest_sha256': PROTOCOL_SHA, 'feature_manifest_sha256': FEATURE_MANIFEST_SHA,
        'foundation_sha256': CHECKPOINT_SHA['v3.5'],
        'train_count': len(train), 'validation_count': len(validation),
        'train_ids': train.record_id.tolist(), 'validation_ids': validation.record_id.tolist(),
        'input_sha256': {part: protocol_manifest['outputs'][prefix + '/' + part + '.csv'] for part in ['train', 'val']},
        'source_commit': '9393b12a46bfc32369a89a53f6c578d22e41faba',
        'code_sha256': {path.name: sha256(path) for path in [Path(__file__), Path(__file__).with_name('common.py'), benchmark_dir/'baselines.py']},
        'package_versions': {name: importlib.metadata.version(name) for name in ['tabpfn','torch','numpy','pandas','scikit-learn','scipy']},
    }
    write_json(args.output/'policy_lock.json', common)
    frozen_dir, tuned_dir = args.output/'tabpfn35_frozen', args.output/'tabpfn35_finetuned'
    frozen_dir.mkdir()
    tuned_dir.mkdir()

    def new_model(weights):
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V3_5, model_path=str(weights), n_estimators=8,
            random_state=args.seed, device='cuda', inference_config=INFERENCE_CONFIG)

    frozen = new_model(checkpoint)
    frozen.fit(train_x, train_y)
    frozen_values = frozen.predict(val_x)
    aligned_predictions(validation, frozen_values).to_csv(frozen_dir/'validation_predictions.csv', index=False)
    save_fitted_tabpfn_model(frozen, frozen_dir/'model.tabpfn_fit')
    frozen_metrics = metrics(val_y, frozen_values)
    del frozen
    gc.collect()
    torch.cuda.empty_cache()

    logger = HistoryLogger(tuned_dir, float(np.var(val_y)))
    logger.frozen_mse = frozen_metrics['mse']
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    fine = audited_finetuner()(
        model_version=ModelVersion.V3_5, epochs=POLICY['epochs'], random_state=args.seed,
        validation_split_ratio=None, early_stopping_patience=POLICY['patience'], min_delta=0.0,
        n_estimators_finetune=2, n_estimators_validation=8, n_estimators_final_inference=8,
        use_fixed_preprocessing_seed=False, save_checkpoint_interval=None,
        extra_regressor_kwargs={'model_path': str(checkpoint)}, experiment_logger=logger,
    )
    fine.fit(train_x, train_y, X_val=val_x, y_val=val_y, output_dir=tuned_dir/'native')
    validation_audit = fine.validation_audit_
    if len(logger.steps) != len(logger.epochs) or len(validation_audit) != len(logger.epochs) + 1:
        raise ValueError('Expected one full-training episode and one validation per epoch')
    if not any(item['sample_max_parameter_change'] > 0 for item in validation_audit):
        raise ValueError('No trainable parameter change detected')
    if not np.isclose(validation_audit[0]['mse'], frozen_metrics['mse'], rtol=1e-5, atol=1e-7):
        raise ValueError('Initial fine-tuning validation differs from frozen control')
    selected = min(validation_audit, key=lambda value: value['mse'])
    tuned_values = fine.predict(val_x)
    tuned_metrics = metrics(val_y, tuned_values)
    if not np.isclose(tuned_metrics['mse'], selected['mse'], rtol=1e-5, atol=1e-7):
        raise ValueError('Returned weights do not reproduce best validation checkpoint')
    aligned_predictions(validation, tuned_values).to_csv(tuned_dir/'validation_predictions.csv', index=False)
    weights = tuned_dir/'selected_weights.pth'
    save_tabpfn_model(fine.finetuned_estimator_, weights)
    selected_parameter_change = float(np.max(np.abs(parameter_sample(fine.finetuned_estimator_) - fine.initial_parameter_sample_)))
    del fine
    gc.collect()
    torch.cuda.empty_cache()
    tuned = new_model(weights)
    tuned.fit(train_x, train_y)
    checkpoint_reload_values = tuned.predict(val_x)
    if not np.allclose(checkpoint_reload_values, tuned_values, rtol=1e-5, atol=1e-6):
        raise ValueError('Weight checkpoint reload changed validation predictions')
    save_fitted_tabpfn_model(tuned, tuned_dir/'model.tabpfn_fit')
    del tuned
    gc.collect()
    torch.cuda.empty_cache()

    # The policy and weights are fixed before either held-out cohort is read.
    test = load_frame(protocol, protocol_manifest, prefix + '/test.csv')
    hela = load_frame(protocol, protocol_manifest, 'hela_full.csv')
    if set(test.record_id) & (set(train.record_id) | set(validation.record_id)):
        raise ValueError('Outer test ID overlap')
    if args.axis == 'grouped':
        for column in ['target_group_id', 'split_group_id']:
            if set(test[column]) & (set(train[column]) | set(validation[column])):
                raise ValueError('Outer test grouped-split overlap')
    if len(hela) != 1047 or not hela.cell_line.str.strip().str.lower().eq('hela').all():
        raise ValueError('Expected full 1047-row HeLa cohort')
    if set(hela.record_id) & (set(train.record_id) | set(validation.record_id) | set(test.record_id)):
        raise ValueError('HeLa record overlap')
    index = []
    for directory, reference, validation_metrics in [(frozen_dir, frozen_values, frozen_metrics), (tuned_dir, tuned_values, tuned_metrics)]:
        model = load_fitted_tabpfn_model(directory/'model.tabpfn_fit', device='cuda')
        reloaded = model.predict(val_x)
        if not np.allclose(reference, reloaded, rtol=1e-5, atol=1e-6):
            raise ValueError('Fitted-state reload changed validation predictions')
        heldout_metrics = {}
        for name, frame in [('test', test), ('hela', hela)]:
            values = model.predict(representation(frame, columns, sequence_features))
            aligned_predictions(frame, values).to_csv(directory/f'{name}_predictions.csv', index=False)
            heldout_metrics[name] = metrics(frame.efficiency.to_numpy(), values)
        metadata = {**common, 'tool': directory.name, 'validation_metrics': validation_metrics,
                    'heldout_metrics': heldout_metrics, 'status': 'complete',
                    'fitted_model_sha256': sha256(directory/'model.tabpfn_fit'),
                    'reload_max_abs_difference': float(np.max(np.abs(reference-reloaded))),
                    'seed_meaning': 'preprocessing/ensemble random_state' if directory == frozen_dir else 'optimization and preprocessing random_state'}
        if directory == tuned_dir:
            metadata.update({'epochs_completed': len(logger.epochs), 'optimizer_steps': len(logger.steps),
                             'selected_epoch': selected['epoch'], 'selected_mse': selected['mse'],
                             'epoch_convention': 'selected_epoch 0 = foundation; positive values = completed epochs; logger train/epoch is zero-based',
                             'selected_parameter_sample_change': selected_parameter_change,
                             'weights_sha256': sha256(weights),
                             'checkpoint_reload_max_abs_difference': float(np.max(np.abs(checkpoint_reload_values-tuned_values)))})
        write_json(directory/'train_meta.json', metadata)
        row = {'tool': directory.name, 'axis': args.axis, 'fold': args.fold,
               'training_seed': args.seed, 'status': 'verified', 'run_dir': str(directory.resolve())}
        for column, filename in [('test_predictions', 'test_predictions.csv'), ('hela_predictions', 'hela_predictions.csv'), ('train_meta', 'train_meta.json')]:
            row[column] = str((directory/filename).resolve())
            row[column+'_sha256'] = sha256(directory/filename)
        index.append(row)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(index).to_csv(args.output/'run_index.csv', index=False)
    write_json(args.output/'complete.json', {'status': 'verified', 'policy_version': POLICY['version'],
               'run_index_sha256': sha256(args.output/'run_index.csv'),
               'elapsed_seconds': time.monotonic()-started,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    print(f'COMPLETE {args.axis} fold={args.fold} seed={args.seed}', flush=True)


if __name__ == '__main__':
    main()
