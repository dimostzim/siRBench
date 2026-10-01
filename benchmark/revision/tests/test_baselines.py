import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / 'baselines.py'
spec = importlib.util.spec_from_file_location('baselines', PATH)
baselines = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baselines)


def test_iscore_matches_published_examples_and_extremes():
    examples = pd.read_csv(baselines.REFERENCE/'iscore_published_examples.csv')
    scores = baselines.iscore(examples.guide_with_overhang.str[:19])
    np.testing.assert_allclose(scores, examples.published_iscore, rtol=0, atol=1e-12)
    coefficients = pd.read_csv(baselines.REFERENCE/'iscore_2007_antisense.csv').set_index('position')
    worst = ''.join(coefficients.idxmin(axis=1))
    best = ''.join(coefficients.idxmax(axis=1))
    np.testing.assert_allclose(baselines.iscore([worst,best]), [0,100], atol=1e-12)


def test_uitei_rules_use_guide_positions_and_gc_stretch():
    passing = 'AUAA'+'G'*5+'A'*9+'C'
    sequences = [passing, 'G'+passing[1:], passing[:-1]+'A', 'A'+'G'*17+'C', 'AUAAAAA'+'G'*10+'AC']
    assert baselines.uitei_functional(sequences).tolist() == [1,0,0,0,0]


def test_sequence_features_exclude_assay_covariates_and_validate_length():
    frame = pd.DataFrame({'siRNA':['A'*19,'C'*19], 'source':['x','y'],'cell_line':['z','w']})
    features = baselines.sequence_features(frame)
    assert features.shape == (2,76)
    np.testing.assert_equal(features.sum(axis=1),19)
    frame.siRNA = ['A'*18,'C'*19]
    with pytest.raises(ValueError,match='19-nt'):
        baselines.sequence_features(frame)


def test_ridge_scaler_fits_training_rows_only():
    train_x = np.array([[0.,1.],[1.,2.],[2.,3.]])
    validation_x = np.array([[100.,101.],[200.,201.]])
    model, selection = baselines.ridge_fit(train_x,validation_x,np.array([0.,.5,1.]),np.array([.4,.8]))
    np.testing.assert_allclose(model['standardscaler'].mean_, [1.,2.])
    assert selection['selected_alpha'] in baselines.ALPHAS
