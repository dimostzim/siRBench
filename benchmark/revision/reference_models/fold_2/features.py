"""Stateless guide/flank CNN tensors and exact unweighted positional encoding."""
from pathlib import Path
import csv
import numpy as np

FLANK_POSITIONS = tuple(range(19)) + tuple(range(38, 57))
FEATURE_NAMES = tuple([f'guide_{p+1:02d}_{b}' for p in range(19) for b in 'ACGU'] +
                      [f'context_{p+1:02d}_{b}' for p in FLANK_POSITIONS for b in 'ACGTX'])
CONFIG = {'version': 4,
          'tensor_shapes_without_batch': {'guide': [4,19], 'left': [5,19], 'right': [5,19], 'positional': [266]},
          'tensor_layout': 'channel-first; positional vector is position-major guide,left,right',
          'left_slice': [0,19], 'right_slice': [38,57],
          'central_target': 'omitted as redundant', 'padding': 'explicit X channel; preserve positions',
          'guide_orientation': 'unchanged antisense 5-to-3', 'guide_alphabet': 'ACGU', 'guide_length': 19,
          'context_alphabet': 'ACGTX', 'context_length': 57,
          'context_positions_zero_based': list(FLANK_POSITIONS),
          'dtype': 'float32', 'dimension': 266, 'active_value': 1.0,
          'scaling': 'none', 'label_transform': 'training-only population standardization',
          'feature_names': list(FEATURE_NAMES)}

def representation_config():
    import copy
    return copy.deepcopy(CONFIG)

def read_inputs(input_dir):
    """Read only input/data.csv interface (argument is the input directory)."""
    with (Path(input_dir) / 'data.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        if not {'id', 'siRNA', 'extended_mRNA'}.issubset(reader.fieldnames or []):
            raise ValueError('Required input columns missing')
        rows = list(reader)
    ids = [r['id'] for r in rows]
    if any(not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Input IDs must be nonempty and unique')
    return ids, rows

def transform(rows):
    """Stateless position-major unweighted one-hot; guide remains antisense."""
    x = np.zeros((len(rows), 266), dtype=np.float32)
    for i, row in enumerate(rows):
        for key, length, alphabet, offset in [('siRNA', 19, 'ACGU', 0),
                                               ('extended_mRNA', 57, 'ACGTX', 76)]:
            seq = row.get(key)
            if not isinstance(seq, str) or len(seq) != length or not set(seq) <= set(alphabet):
                raise ValueError(f'Invalid {key} at input row {i}: expected length {length}, alphabet {alphabet}')
            positions = range(19) if key == 'siRNA' else FLANK_POSITIONS
            for j, p in enumerate(positions):
                x[i, offset + j * len(alphabet) + alphabet.index(seq[p])] = 1.0
    return x

def transform_tensors(rows):
    """Return independent contiguous N,C,L arrays and N,266 linear features."""
    positional = transform(rows)
    n = len(rows)
    return {
        'guide': np.ascontiguousarray(positional[:, :76].reshape(n, 19, 4).transpose(0, 2, 1)),
        'left': np.ascontiguousarray(positional[:, 76:171].reshape(n, 19, 5).transpose(0, 2, 1)),
        'right': np.ascontiguousarray(positional[:, 171:].reshape(n, 19, 5).transpose(0, 2, 1)),
        'positional': positional,
    }


def load_tensors(input_dir):
    ids, rows = read_inputs(input_dir)
    return ids, transform_tensors(rows)


def load_features(input_dir):
    ids, rows = read_inputs(input_dir)
    return ids, transform(rows)

def load_labels(labels_path, ids):
    """Explicit one-to-one ID alignment; labels never used by transform."""
    with Path(labels_path).open(newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        column = 'numeric_label' if 'numeric_label' in fields else 'label'
        if 'id' not in fields or column not in fields:
            raise ValueError('Required label columns missing')
        rows = list(reader)
    keyed = {r['id']: r[column] for r in rows}
    if len(keyed) != len(rows) or len(ids) != len(set(ids)) or set(keyed) != set(ids):
        raise ValueError('Labels and input IDs must have a one-to-one exact match')
    y = np.asarray([float(keyed[i]) for i in ids], dtype=np.float64)
    if not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
        raise ValueError('Labels must be finite and in [0,1]')
    return y


def fit_target_transform(training_y):
    """Call exclusively with complete training labels; serialize returned state."""
    y = np.asarray(training_y, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all():
        raise ValueError('Expected nonempty finite training target vector')
    mean, std = float(y.mean()), float(y.std(ddof=0))
    return {'mean': mean, 'population_std': std, 'scale': std if std > 0 else 1.0}


def transform_targets(y, state):
    return ((np.asarray(y, dtype=np.float64) - state['mean']) / state['scale']).astype(np.float32)


def inverse_targets(z, state):
    return np.asarray(z, dtype=np.float64) * state['scale'] + state['mean']
