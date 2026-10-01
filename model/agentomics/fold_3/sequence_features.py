"""Stateless CNN sequence tensors. Loading helpers adapted from iteration 3 only."""
from pathlib import Path
import numpy as np
import pandas as pd

def read_inputs(input_dir):
    frame = pd.read_csv(Path(input_dir) / 'data.csv', dtype=str, keep_default_na=False)
    required = {'id', 'siRNA', 'extended_mRNA'}
    if not required.issubset(frame.columns):
        raise ValueError('Input requires id, siRNA, extended_mRNA')
    if frame['id'].eq('').any() or frame['id'].duplicated().any():
        raise ValueError('IDs must be nonempty and unique')
    return frame

def read_labels(labels_path, ids):
    """Training/evaluation helper only; never needed by inference."""
    labels = pd.read_csv(labels_path, dtype={'id': str}, keep_default_na=False)
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

DESCRIPTOR_NAMES = ['guide_GC_fraction', 'guide_GC_fraction_squared'] + [
    name for k in (2,3,4,5) for name in
    (f'guide_first_{k}_AU', f'guide_last_{k}_AU', f'guide_first_minus_last_{k}_AU')]
REPRESENTATION = {
    'version': 'separate_flank_cnn_v1', 'dtype': 'float32',
    'outputs': {'guide': [4,19], 'left': [5,19], 'right': [5,19], 'descriptors': [14]},
    'batch_axis': 0, 'sequence_axes': ['channel','position'],
    'guide_channels': list('ACGU'), 'context_channels': list('ACGTX'),
    'guide_orientation': 'supplied antisense 5-prime to 3-prime, unchanged',
    'left_slice': [0,19], 'right_slice': [38,57], 'central_segment': 'excluded; checked for consistency only',
    'descriptor_names': DESCRIPTOR_NAMES,
    'descriptor_definition': 'GC=count(G or C)/19; terminal AU=count(A or U)/k; difference=first-last',
    'flatten_order': 'C order: channel-major, then increasing position',
    'direct_branch_order': 'flattened guide (76), then descriptors (14)',
    'padding': 'X explicit channel; outside-sequence convolution padding all-zero',
    'normalization': None, 'learned_preprocessing': None, 'label_transform': None,
    'empty_input': 'return zero-row arrays with unchanged trailing dimensions',
}

def validate_sequences(frame):
    for col, alphabet, length in [('siRNA','ACGU',19), ('extended_mRNA','ACGTX',57)]:
        if col not in frame:
            raise ValueError(f'Missing input column: {col}')
        for row, seq in enumerate(frame[col]):
            if not isinstance(seq,str) or len(seq)!=length or set(seq)-set(alphabet):
                raise ValueError(f'Invalid {col} at row {row}')
    complement=str.maketrans('ACGU','TGCA')
    for row,(g,c) in enumerate(zip(frame.siRNA, frame.extended_mRNA)):
        if c[19:38] != g.translate(complement)[::-1]:
            raise ValueError(f'Central target inconsistency at row {row}')

def transform(frame):
    """Return independent float32 arrays; no ID access, fitted state or batch statistics."""
    validate_sequences(frame)
    out={name:np.zeros((len(frame),*shape),dtype=np.float32)
         for name,shape in REPRESENTATION['outputs'].items()}
    for row,(guide,context) in enumerate(zip(frame.siRNA,frame.extended_mRNA)):
        for key,seq,alphabet in [('guide',guide,'ACGU'),('left',context[:19],'ACGTX'),('right',context[38:57],'ACGTX')]:
            out[key][row,[alphabet.index(b) for b in seq],np.arange(19)]=1
        gc=sum(b in 'GC' for b in guide)/19
        desc=[gc,gc*gc]
        for k in (2,3,4,5):
            first=sum(b in 'AU' for b in guide[:k])/k
            last=sum(b in 'AU' for b in guide[-k:])/k
            desc.extend([first,last,first-last])
        out['descriptors'][row]=desc
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError('Nonfinite features')
    return out
