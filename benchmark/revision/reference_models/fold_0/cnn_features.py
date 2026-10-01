"""Deterministic, query-independent sequence representation. No learned input state."""
from pathlib import Path
import numpy as np
import pandas as pd

SCHEMA = {
    'version': 1, 'dtype': 'float32', 'layout': 'N,C,L',
    'guide': {'column': 'siRNA', 'channels': list('ACGU'), 'shape': [4,19], 'orientation': 'supplied antisense 5prime-to-3prime'},
    'left': {'column': 'extended_mRNA', 'slice': [0,19], 'channels': list('ACGTX'), 'shape': [5,19]},
    'right': {'column': 'extended_mRNA', 'slice': [38,57], 'channels': list('ACGTX'), 'shape': [5,19]},
    'positional': {'shape': [266], 'order': ['guide','left','right'], 'flatten': 'C-order: channel then position'},
    'central_context': 'validate reverse complement, omit as redundant',
    'unknown': 'X is explicit context channel; reject all other unexpected symbols',
    'boundary_padding': 'all-zero convolution padding, distinct from X',
    'label_transform': 'training-only mean and population std; (y-mean)/std; inverse mean+std*z; no clipping',
    'excluded': ['id','row order','metadata'],
}

def _ids(frame):
    if 'id' not in frame or not frame.id.map(lambda x: isinstance(x,str) and bool(x)).all() or frame.id.duplicated().any():
        raise ValueError('IDs must be unique nonempty strings')

def read_inputs(input_dir):
    frame = pd.read_csv(Path(input_dir)/'data.csv', dtype=str, keep_default_na=False)
    _ids(frame)
    return frame

def encode(frame):
    """Return guide, left, right, positional float32 arrays, including empty inputs."""
    if not {'siRNA','extended_mRNA'}.issubset(frame.columns):
        raise ValueError('Missing sequence columns')
    n = len(frame)
    guide = np.zeros((n,4,19), dtype=np.float32)
    left = np.zeros((n,5,19), dtype=np.float32)
    right = np.zeros((n,5,19), dtype=np.float32)
    for i,(g,c) in enumerate(zip(frame.siRNA,frame.extended_mRNA)):
        if not isinstance(g,str) or len(g)!=19 or set(g)-set('ACGU'):
            raise ValueError(f'Invalid guide at input position {i}')
        if not isinstance(c,str) or len(c)!=57 or set(c)-set('ACGTX'):
            raise ValueError(f'Invalid context at input position {i}')
        if c[19:38] != g.translate(str.maketrans('ACGU','TGCA'))[::-1]:
            raise ValueError(f'Central reverse-complement mismatch at input position {i}')
        for target,seq,alphabet in [(guide,g,'ACGU'),(left,c[:19],'ACGTX'),(right,c[38:],'ACGTX')]:
            target[i,[alphabet.index(b) for b in seq],np.arange(19)] = 1
    positional = np.concatenate([guide.reshape(n,76),left.reshape(n,95),right.reshape(n,95)],axis=1)
    return guide,left,right,positional

def read_labels(split_dir, inputs):
    _ids(inputs)
    labels = pd.read_csv(Path(split_dir)/'labels.csv',dtype={'id':str},keep_default_na=False)
    _ids(labels)
    if set(labels.id)!=set(inputs.id):
        raise ValueError('Input and label ID sets differ')
    column = 'numeric_label' if 'numeric_label' in labels else 'label'
    y = pd.to_numeric(labels.set_index('id').loc[inputs.id,column],errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(y).all() or ((y<0)|(y>1)).any():
        raise ValueError('Labels must be finite in [0,1]')
    return y

def fit_label_stats(training_y):
    y = np.asarray(training_y,dtype=np.float64)
    if y.ndim!=1 or not y.size or not np.isfinite(y).all():
        raise ValueError('Need nonempty finite training targets')
    std = float(y.std(ddof=0))
    return {'mean':float(y.mean()),'std':std if std>0 else 1.0,'population_std':std,'constant_label_fallback':std==0}

def standardize(y, stats):
    return ((np.asarray(y)-stats['mean'])/stats['std']).astype(np.float32)

def inverse_targets(z, stats):
    return np.asarray(z,dtype=np.float64)*stats['std']+stats['mean']
