"""Training-only global feature normalization for polynomial KRR."""
import numpy as np
from sequence_features import encode, feature_schema, WIDTH

def fit_normalization(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != WIDTH or not len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid training feature matrix')
    mu = x.mean(axis=0)
    s2 = float(np.mean(np.sum((x-mu)**2, axis=1)))
    if not np.isfinite(s2) or s2 <= 0:
        raise ValueError('Global training variance must be positive')
    return mu, s2

def transform(x, mu, s2):
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != WIDTH or mu.shape != (WIDTH,):
        raise ValueError('Feature schema mismatch')
    if not np.isfinite(x).all() or not np.isfinite(mu).all() or not np.isfinite(s2) or s2 <= 0:
        raise ValueError('Invalid normalization input/state')
    return (x-mu)/np.sqrt(s2)

def representation_schema():
    schema = feature_schema(w_flank=0.25)
    schema['normalization'] = {'dtype':'float64', 'fit':'training only',
        'mu':'mean(X_train, axis=0)',
        's2':'mean(sum((X_train-mu)**2, axis=1))',
        'transform':'(X-mu)/sqrt(s2)',
        'per_column_scaling':False, 'per_row_scaling':False}
    return schema
