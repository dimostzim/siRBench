"""Portable three-view interface; no fitted state created on import."""
import numpy as np
import sequence_features as sequence
import thermo_features as physical
import thermo_cnn_features as thermo_cnn
import tree_base_features as tree_base
from sequence_features import read_inputs, read_labels
from thermo_cnn_features import ThermoScaler

def raw_transform(frame):
    cnn = sequence.transform(frame)
    thermo = physical.transform(frame)
    tree = np.concatenate([tree_base.transform(frame), thermo], axis=1)
    return {'sequence': cnn, 'thermo_raw': thermo, 'tree': tree}

def transform(frame, scaler):
    raw = raw_transform(frame)
    augmented = dict(raw['sequence'])
    augmented['thermo'] = scaler.transform(raw['thermo_raw'])
    return {'sequence': raw['sequence'], 'thermo_cnn': augmented, 'tree': raw['tree']}
