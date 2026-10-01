"""Synthetic-only checks for the expanded benchmark renderer."""
import itertools
import json
import sys

import pandas as pd
import pytest

import render_benchmark_results as renderer


def intervals():
    rows = []
    for tool, cohort, metric in itertools.product(
            set(renderer.LABELS) | renderer.HISTORICAL, renderer.COHORTS, renderer.METRICS):
        rows.append({'tool': tool, 'cohort': cohort, 'metric': metric,
                     'comparison_role': 'historical_reference' if tool in renderer.HISTORICAL else 'primary_comparator',
                     'estimate': .4, 'ci_lower': .2, 'ci_upper': .6, 'finite_draws': 2000,
                     'bootstrap_draws': 2000, 'clusters': 43 if cohort == 'test' else 45})
    return pd.DataFrame(rows)


def test_both_complete_matrices_required():
    frame = pd.DataFrame(itertools.product(renderer.TABPFN_TOOLS, ['grouped', 'random'], range(5), range(3)),
                         columns=['tool', 'axis', 'fold', 'training_seed']).assign(status='verified')
    renderer.validate_index(frame, renderer.TABPFN_TOOLS)
    with pytest.raises(ValueError, match='Incomplete'):
        renderer.validate_index(frame.iloc[:-1], renderer.TABPFN_TOOLS)
    with pytest.raises(ValueError, match='duplicate'):
        renderer.validate_index(pd.concat([frame, frame.iloc[:1]]), renderer.TABPFN_TOOLS)
    with pytest.raises(ValueError, match='Unverified'):
        renderer.validate_index(frame.assign(status='pending'), renderer.TABPFN_TOOLS)


def test_intervals_require_all_methods_and_retain_historical_role():
    frame = intervals()
    renderer.validate_intervals(frame)
    with pytest.raises(ValueError, match='fifteen'):
        renderer.validate_intervals(frame[frame.tool.ne('tabpfn35_finetuned')])
    frame.loc[frame.tool.eq('iscore_2007_fixed'), 'comparison_role'] = 'primary_comparator'
    with pytest.raises(ValueError, match='historical'):
        renderer.validate_intervals(frame)


def test_invalid_bootstrap_bounds_and_groups_rejected():
    frame = intervals()
    with pytest.raises(ValueError, match='clusters'):
        renderer.validate_intervals(frame.assign(clusters=100))
    with pytest.raises(ValueError, match='confidence interval'):
        renderer.validate_intervals(frame.assign(ci_lower=.7))


def test_undefined_metrics_preserved_in_tables(tmp_path):
    frame = intervals()
    selected = frame.tool.eq('training_mean') & frame.cohort.eq('hela_full') & frame.metric.eq('pearson_r')
    frame.loc[selected, ['estimate', 'ci_lower', 'ci_upper']] = float('nan')
    frame.loc[selected, 'finite_draws'] = 0
    renderer.validate_intervals(frame)
    renderer.write_tables(frame, tmp_path)
    text = (tmp_path/'hela_full_metrics.md').read_text()
    assert '| Training mean | -- |' in text
    assert 'TabPFN-3.5 (frozen)' in text and 'TabPFN-3.5 (fine-tuned)' in text
    assert 'iscore_2007' not in text
    historical = pd.read_csv(tmp_path/'historical_reference_metrics.csv')
    assert set(historical.tool) == renderer.HISTORICAL


def test_synthetic_plot_exports_all_formats(tmp_path):
    renderer.plot_metrics(intervals(), tmp_path)
    for extension in ['pdf', 'svg', 'png']:
        assert (tmp_path/f'primary_performance.{extension}').stat().st_size > 1000


def test_synthetic_full_render_checks_sealed_indexes(tmp_path, monkeypatch):
    analysis = tmp_path/'analysis'
    analysis.mkdir()
    intervals().to_csv(analysis/'primary_group_bootstrap.csv', index=False)
    indexes = []
    for name, tools in [('original', renderer.ORIGINAL_TOOLS), ('tabpfn', renderer.TABPFN_TOOLS)]:
        index = tmp_path/f'{name}.csv'
        pd.DataFrame(itertools.product(tools, ['grouped', 'random'], range(5), range(3)),
                     columns=['tool', 'axis', 'fold', 'training_seed']).assign(status='verified').to_csv(index, index=False)
        indexes.append(index)
    manifest = {'methods': sorted(set(renderer.LABELS) | renderer.HISTORICAL),
                'bootstrap_draws': 2000, 'bootstrap_seed': 20260921,
                'inputs': {str(path): renderer.sha256(path) for path in indexes}}
    (analysis/'analysis_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(renderer, 'ORIGINAL_INDEX_SHA', renderer.sha256(indexes[0]))
    output = tmp_path/'output'
    arguments = ['renderer', '--analysis', str(analysis), '--original-index', str(indexes[0]),
                 '--tabpfn-index', str(indexes[1]), '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', arguments)
    renderer.main()
    result = json.loads((output/'manifest.json').read_text())
    assert result['inputs'][str(indexes[1])] == renderer.sha256(indexes[1])
    assert set(pd.read_csv(output/'primary_metrics.csv').tool) == set(renderer.LABELS)
    indexes[1].write_text(indexes[1].read_text()+'\n')
    monkeypatch.setattr(sys, 'argv', arguments[:-1]+[str(tmp_path/'changed-output')])
    with pytest.raises(ValueError, match='sealed index'):
        renderer.main()
