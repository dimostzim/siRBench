"""Iteration-17 postprocessing: fit on honest training-only OOF means only."""
import numpy as np

def fit_oof_affine(oof_mean, training_labels):
    m = np.asarray(oof_mean, dtype=np.float64)
    y = np.asarray(training_labels, dtype=np.float64)
    if m.ndim != 1 or y.shape != m.shape or not m.size or not np.isfinite(m).all() or not np.isfinite(y).all():
        raise ValueError('Expected complete finite training OOF predictions and labels')
    centered = m-m.mean()
    denominator = np.dot(centered, centered)
    a = float(np.dot(centered, y-y.mean())/denominator) if denominator > 1e-12 else 0.0
    return {'a': a, 'b': float(y.mean()-a*m.mean())}

def apply_affine(mean_prediction, calibration):
    m = np.asarray(mean_prediction, dtype=np.float64)
    result = float(calibration['a'])*m + float(calibration['b'])
    if m.ndim != 1 or not np.isfinite(result).all():
        raise ValueError('Expected finite one-dimensional predictions')
    return result
