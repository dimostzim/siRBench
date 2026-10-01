"""Feature preparation and file helpers for the TabPFN models."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


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

def aligned_predictions(frame, values):
    values = np.asarray(values, dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError('Predictions must be finite and aligned with validation rows')
    return pd.DataFrame({'record_id': frame.record_id, 'label': frame.efficiency,
                         'pred_label': values})
