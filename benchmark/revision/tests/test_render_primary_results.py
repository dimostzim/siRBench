import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render_primary_results import COHORTS, HISTORICAL, LABELS, METRICS, interval_text, validate_intervals


def interval_fixture():
    return pd.DataFrame([{'tool': tool, 'cohort': cohort, 'metric': metric,
        'comparison_role': 'historical_reference' if tool in HISTORICAL else 'primary_comparator',
        'estimate': .2, 'ci_lower': .1, 'ci_upper': .3, 'finite_draws': 2000,
        'bootstrap_draws': 2000, 'clusters': 43 if cohort == 'test' else 45}
        for tool, cohort, metric in itertools.product(set(LABELS) | HISTORICAL, COHORTS, METRICS)])


@pytest.mark.parametrize('corruption', ['missing_method', 'duplicate', 'wrong_role', 'wrong_clusters', 'reversed_interval'])
def test_incomplete_or_inconsistent_analysis_rejected(corruption):
    frame = interval_fixture()
    if corruption == 'missing_method': frame = frame[frame.tool.ne('ensirna')]
    elif corruption == 'duplicate': frame = pd.concat([frame, frame.iloc[[0]]])
    elif corruption == 'wrong_role': frame.loc[frame.tool.eq('iscore_2007_fixed'), 'comparison_role'] = 'primary_comparator'
    elif corruption == 'wrong_clusters': frame.loc[0, 'clusters'] = 47
    else: frame.loc[0, 'ci_lower'] = .4
    with pytest.raises(ValueError): validate_intervals(frame)


def test_undefined_metric_is_not_displayed_as_zero():
    frame = interval_fixture()
    frame.loc[0, ['estimate','ci_lower','ci_upper']] = np.nan
    frame.loc[0, 'finite_draws'] = 0
    validate_intervals(frame)
    assert interval_text(frame.iloc[0]) == '--'
    assert interval_text(frame.iloc[1]) == '0.200 [0.100, 0.300]'


def test_panels_use_correct_fields_and_keep_defined_oof_mean_baseline(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from render_primary_results import plot_metrics

    frame = interval_fixture()
    mean_test = frame.tool.eq('training_mean') & frame.cohort.eq('test') & frame.metric.eq('pearson_r')
    frame.loc[mean_test, ['estimate','ci_lower','ci_upper']] = [-.096,-.283,.104]
    mean_hela = frame.tool.eq('training_mean') & frame.cohort.eq('hela_full') & frame.metric.eq('pearson_r')
    frame.loc[mean_hela, ['estimate','ci_lower','ci_upper']] = np.nan
    frame.loc[mean_hela, 'finite_draws'] = 0
    # Distinct sentinels detect a cohort/metric panel swap.
    for metric, cohort, value in [('pearson_r','test',.61), ('pearson_r','hela_full',.52),
                                  ('r2','test',.34), ('r2','hela_full',.23)]:
        selected = frame.tool.eq('attsioff') & frame.cohort.eq(cohort) & frame.metric.eq(metric)
        frame.loc[selected, ['estimate','ci_lower','ci_upper']] = [value,value-.05,value+.05]
    # Historical rows must never appear as extra plotted methods.
    frame.loc[frame.tool.isin(HISTORICAL),'estimate'] = .999
    closed = []
    real_close = plt.close
    monkeypatch.setattr(plt,'close',lambda figure:closed.append(figure))
    try:
        plot_metrics(frame,tmp_path)
        figure = closed[-1]
        axes = figure.axes
        point_values = [[float(line.get_xdata()[0]) for line in axis.lines if line.get_marker()=='o'] for axis in axes]
        assert -.096 in point_values[0]
        assert len(point_values[0]) == len(LABELS)
        assert len(point_values[1]) == len(LABELS)-1
        for values,sentinel in zip(point_values,[.61,.52,.34,.23]):
            assert sentinel in values
            assert .999 not in values
        assert any(text.get_text()=='undefined' for text in axes[1].texts)
    finally:
        for figure in closed:
            real_close(figure)
