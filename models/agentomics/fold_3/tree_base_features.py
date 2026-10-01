"""Stateless positional sequence representation; no fitted preprocessing."""
from pathlib import Path
import numpy as np
import pandas as pd

REPRESENTATION = {
    'version': 'positional_onehot_v1',
    'guide_channels': list('ACGU'), 'guide_length': 19,
    'context_channels': list('ACGTX'), 'context_length': 57,
    'ordering': 'guide then context, position-major, zero-based',
    'dimension': 361, 'dtype': 'float32',
    'normalization': None, 'label_transform': None,
}

def read_inputs(input_dir):
    frame = pd.read_csv(Path(input_dir) / 'data.csv', dtype=str, keep_default_na=False)
    required = {'id', 'siRNA', 'extended_mRNA'}
    if not required.issubset(frame.columns):
        raise ValueError('Input requires id, siRNA, extended_mRNA')
    if frame['id'].eq('').any() or frame['id'].duplicated().any():
        raise ValueError('IDs must be nonempty and unique')
    return frame

def positional_transform(frame):
    """Encode rows independently; identifiers never enter the feature matrix."""
    result = np.zeros((len(frame), 361), dtype=np.float32)
    for column, alphabet, length, offset in [('siRNA', 'ACGU', 19, 0),
                                            ('extended_mRNA', 'ACGTX', 57, 76)]:
        if column not in frame:
            raise ValueError(f'Missing input column: {column}')
        mapping = {base: i for i, base in enumerate(alphabet)}
        for row, sequence in enumerate(frame[column]):
            if not isinstance(sequence, str) or len(sequence) != length or set(sequence) - set(alphabet):
                raise ValueError(f'Invalid {column} at input row {row}: expected length {length}, alphabet {alphabet}')
            indices = offset + np.arange(length) * len(alphabet) + np.array([mapping[b] for b in sequence])
            result[row, indices] = 1.0
    return result

def read_labels(labels_path, ids):
    """Training/evaluation helper only; never needed by inference."""
    labels = pd.read_csv(labels_path, dtype={'id': str})
    column = 'numeric_label' if 'numeric_label' in labels else 'label'
    if column not in labels or 'id' not in labels:
        raise ValueError('Missing label schema')
    ids = list(ids)
    if labels['id'].duplicated().any() or len(set(ids)) != len(ids) or set(labels['id']) != set(ids):
        raise ValueError('Labels and inputs must have identical unique ID membership')
    y = labels.set_index('id').loc[ids, column].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
        raise ValueError('Expected finite continuous labels in [0,1]')
    return y

def feature_names():
    return ([f'guide_{p}_{b}' for p in range(19) for b in 'ACGU'] +
            [f'context_{p}_{b}' for p in range(57) for b in 'ACGTX'])

# Preserve the previous positional names and values as the first 361 columns.
positional_feature_names = feature_names
from itertools import product
import math

KMERS = {k: [''.join(x) for x in product('ACGT', repeat=k)] for k in (2, 3)}

def longest_run(sequence, allowed):
    best = current = 0
    for base in sequence:
        current = current + 1 if base in allowed else 0
        best = max(best, current)
    return best

def region_features(sequence, prefix):
    values = {}
    def add(name, value):
        values[f'{prefix}_{name}'] = value
    counts = {b: sequence.count(b) for b in 'ACGTX'}
    known = sum(counts[b] for b in 'ACGT')
    for b in 'ACGTX':
        add(f'count_{b}', counts[b])
    add('known_fraction', known / len(sequence))
    fractions = {b: counts[b] / known if known else 0.0 for b in 'ACGT'}
    for b in 'ACGT':
        add(f'known_fraction_{b}', fractions[b])
    add('GC_fraction', fractions['G'] + fractions['C'])
    add('AU_fraction', fractions['A'] + fractions['T'])
    add('entropy_bits', -sum(p * math.log2(p) for p in fractions.values() if p))
    for b in 'ACGT':
        add(f'longest_run_{b}', longest_run(sequence, b))
    for k in (2, 3):
        windows = [sequence[i:i+k] for i in range(len(sequence)-k+1) if 'X' not in sequence[i:i+k]]
        for word in KMERS[k]:
            add(f'kmer_{word}_frequency', windows.count(word) / len(windows) if windows else 0.0)
        add(f'kmer_{k}_eligible_count', len(windows))
    return values

def engineered_features(guide, context):
    guide = guide.replace('U', 'T')
    values = {}
    for prefix, region in [('guide_summary', guide), ('left_flank', context[:19]), ('right_flank', context[38:57])]:
        values.update(region_features(region, prefix))
    for k in (3, 5, 7):
        for i in range(20-k):
            values[f'guide_GC_window_{k}_start_{i}'] = sum(b in 'GC' for b in guide[i:i+k]) / k
    for k in (2, 3, 4, 5):
        first = sum(b in 'AT' for b in guide[:k]) / k
        last = sum(b in 'AT' for b in guide[-k:]) / k
        values[f'guide_first_{k}_AU'] = first
        values[f'guide_last_{k}_AU'] = last
        values[f'guide_first_minus_last_{k}_AU'] = first-last
    values['guide_GC_fraction_squared'] = (sum(b in 'GC' for b in guide)/19)**2
    values['guide_longest_AU_run'] = longest_run(guide, 'AT')
    values['guide_longest_GC_run'] = longest_run(guide, 'GC')
    for i in range(18):
        for word in KMERS[2]:
            values[f'guide_dinucleotide_{i}_{word}'] = float(guide[i:i+2] == word)
    return values

ENGINEERED_NAMES = list(engineered_features('A'*19, 'A'*57))

def feature_names():
    return positional_feature_names() + ENGINEERED_NAMES

def transform(frame):
    positional = positional_transform(frame)  # validates alphabets before descriptors
    engineered = np.asarray([list(engineered_features(g, c).values())
                             for g, c in zip(frame['siRNA'], frame['extended_mRNA'])], dtype=np.float32)
    engineered = engineered.reshape(len(frame), len(ENGINEERED_NAMES))
    result = np.concatenate([positional, engineered], axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite sequence feature')
    return result

REPRESENTATION.update({
    'version': 'positional_plus_sequence_descriptors_v2',
    'dimension': len(feature_names()),
    'feature_names': feature_names(),
    'ordering': '361 unchanged positional features; guide/left/right summaries; guide GC windows; terminal AU; GC squared; AU/GC runs; positional dinucleotides. Exact names are authoritative.',
    'engineered_alphabet': 'ACGT; guide U converted to T only for engineered descriptors',
    'regions': {'guide': 'all 19', 'left_flank': '[0:19]', 'right_flank': '[38:57]'},
    'unknown_policy': 'X retains position and breaks runs and kmer windows; explicitly counted; excluded from known-base composition denominator',
    'zero_denominators': 'All composition and kmer frequencies are zero when their denominator is zero; entropy is zero without known bases.',
    'entropy': 'Shannon base-2 entropy over A/C/G/T; zero terms omitted',
    'kmer_order': 'lexicographic ACGT, k=2 then k=3; eligible window count follows each frequency block',
    'fitted_preprocessing': False,
})
