"""Validate prediction-artifact identities and read any supplied sealed bytes."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ARTIFACT_COLUMNS = ('test_predictions', 'hela_predictions', 'train_meta')


def validate_artifact_paths(index):
    identities = {}
    for column in ARTIFACT_COLUMNS:
        if column not in index or index[column].isna().any():
            raise ValueError(f'Missing primary artifact column: {column}')
        for row in index.itertuples(index=False):
            path = Path(getattr(row, column)).resolve()
            if not path.is_file():
                raise ValueError(f'Missing primary artifact file: {path}')
            identity = (row.tool, row.axis, row.fold, row.training_seed, column)
            if path in identities and identities[path] != identity:
                raise ValueError(f'Artifact path reused across run identities: {path}')
            identities[path] = identity


def read_artifact_bytes(row, column):
    path = Path(getattr(row, column))
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    sealed_column = column + '_sha256'
    if hasattr(row, sealed_column):
        expected = getattr(row, sealed_column)
        if pd.isna(expected) or expected != digest:
            raise ValueError(f'Artifact SHA256 mismatch: {path}')
    if column == 'train_meta':
        metadata = json.loads(payload)
        seeds = [metadata['seed']] if 'seed' in metadata else []
        for section in ('config', 'configuration'):
            if section in metadata and 'seed' in metadata[section]:
                seeds.append(metadata[section]['seed'])
        if not seeds:
            raise ValueError(f'Missing recorded training seed: {path}')
        if any(seed != row.training_seed for seed in seeds):
            raise ValueError(f'Training seed differs from index identity: {path}')
    return payload, digest
