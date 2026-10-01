"""Focused checks with synthetic values only; never publish fixture outputs."""
import itertools
import json
import sys

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection

import render_expanded_supplement as renderer


def ranks():
    rows = []
    for axis, fold, seed in itertools.product(['grouped', 'random'], range(5), range(3)):
        for position, tool in enumerate(renderer.NAMES):
            rows.append({'tool': tool, 'axis': axis, 'fold': fold, 'training_seed': seed,
                         'comparison_role': 'historical_reference' if tool in renderer.HISTORICAL else 'primary_comparator',
                         'pearson_r': np.nan if tool == 'training_mean' else position / 20,
                         'r2': position / 25})
    frame = pd.DataFrame(rows)
    eligible = frame[frame.tool.isin(renderer.LABELS)]
    for metric in ['pearson_r', 'r2']:
        frame[metric + '_rank'] = eligible.groupby(['axis', 'fold', 'training_seed'])[metric].rank(ascending=False)
    return frame


def test_rank_pool_requires_both_variants_and_recomputes_ranks():
    frame = ranks()
    renderer.validate_ranks(frame)
    with pytest.raises(ValueError, match='Incomplete'):
        renderer.validate_ranks(frame[frame.tool.ne('tabpfn35_frozen')])
    changed = frame.copy()
    changed.loc[changed.tool.eq('tabpfn35_finetuned'), 'pearson_r_rank'] += 1
    with pytest.raises(ValueError, match='disagree'):
        renderer.validate_ranks(changed)


def test_rank_pool_preserves_undefined_constant_and_historical_exclusion():
    frame = ranks()
    changed = frame.copy()
    changed.loc[changed.tool.eq('training_mean'), 'pearson_r'] = 0
    with pytest.raises(ValueError, match='Constant'):
        renderer.validate_ranks(changed)
    changed = frame.copy()
    changed.loc[changed.tool.eq('iscore_2007_fixed'), 'r2_rank'] = 1
    with pytest.raises(ValueError, match='Historical'):
        renderer.validate_ranks(changed)


def test_paired_direction_reverses_estimate_and_bounds_together():
    frame = pd.DataFrame([{'left': 'guide_thermodynamic_ridge', 'right': 'tabpfn35_frozen',
                           'cohort': 'test', 'metric': 'r2', 'estimate_left_minus_right': -.03,
                           'ci_lower_left_minus_right': -.08, 'ci_upper_left_minus_right': .01}])
    assert renderer.paired_cell(frame, 'tabpfn35_frozen', 'guide_thermodynamic_ridge', 'test', 'r2') == '+0.030 [-0.010, +0.080]'
    with pytest.raises(ValueError, match='Missing or duplicate'):
        renderer.paired_cell(pd.concat([frame, frame]), 'tabpfn35_frozen', 'guide_thermodynamic_ridge', 'test', 'r2')
    frame['estimate_left_minus_right'] = np.nan
    assert renderer.paired_cell(frame, 'tabpfn35_frozen', 'guide_thermodynamic_ridge', 'test', 'r2') == '--'


def test_small_seed_sd_is_not_displayed_as_zero():
    assert renderer.sd_num(6.930148993474079e-5) == r'$6.93\times 10^{-5}$'
    assert renderer.sd_num(0.00011795330822044731) == r'$1.18\times 10^{-4}$'
    assert renderer.sd_num(0) == '0.000'
    assert renderer.sd_num(np.nan) == '--'


def source_summary():
    rows = []
    for cohort, source in renderer.SOURCES:
        for tool in ['guide_thermodynamic_ridge'] + renderer.TABPFN:
            rows.append({'tool': tool, 'axis': 'grouped', 'cohort': cohort,
                         'stratum': 'source', 'stratum_value': source, 'n_min': 44, 'n_max': 44,
                         'pearson_r_mean': .125, 'r2_mean': -12.25})
    return pd.DataFrame(rows)


def test_response_continuation_is_narrow_and_retains_negative_values(tmp_path):
    path = tmp_path / 'continuation.md'
    renderer.write_source_markdown(source_summary(), path)
    text = path.read_text()
    assert '**C. Pearson r**' in text and '**D. R²**' in text
    assert 'Retain the original seven-method panels A and B unchanged' in text
    assert 'TabPFN-3.5 (frozen)' in text and 'TabPFN-3.5 (fine-tuned)' in text
    assert '-12.250' in text
    table_lines = [line for line in text.splitlines() if line.startswith('|')]
    assert len(table_lines) == 26
    assert all(line.count('|') == 6 for line in table_lines)


def test_source_table_rejects_cohort_mismatch():
    frame = source_summary()
    frame.loc[0, 'n_max'] = 43
    with pytest.raises(ValueError, match='cohorts differ'):
        renderer.write_source_markdown(frame, None)


def test_latex_continuation_manifest_matches_source_and_every_cell(tmp_path):
    summary = source_summary()
    summary_path = tmp_path / 'metric_summary.csv'
    summary.to_csv(summary_path, index=False)
    renderer.write_source_latex(summary, summary_path, tmp_path)
    asset = tmp_path / 'Table_R1_continuation.tex'
    text = asset.read_text()
    manifest = json.loads((tmp_path / 'response_source_continuation.json').read_text())
    assert manifest['source_sha256'] == renderer.sha256(summary_path)
    assert manifest['sha256'] == renderer.sha256(asset)
    assert manifest['methods'] == ['guide_thermodynamic_ridge'] + renderer.TABPFN
    assert manifest['comment_id'] == 'R2.m5' and len(manifest['rows']) == 22
    assert 'C. Pearson $r$' in text and 'D. $R^2$' in text
    for row in manifest['rows']:
        expected = ['0.125'] * 3 if row['metric'] == 'pearson_r' else ['-12.250'] * 3
        assert row['values'] == expected
        assert ' & '.join([row['label'], str(row['n']), *expected]) in text
    assert r'\input{Table_R1_continuation.tex}' in (tmp_path / 'Table_R1_continuation_standalone.tex').read_text()


def synthetic_analysis(directory):
    """Complete rendering schema with deliberately synthetic, non-scientific values."""
    directory.mkdir()
    intervals = []
    for tool, cohort, metric in itertools.product(renderer.NAMES, ['test', 'hela_full', 'hela_aligned'], renderer.METRICS):
        intervals.append({'tool': tool, 'cohort': cohort, 'metric': metric, 'estimate': .2,
                          'ci_lower': .1, 'ci_upper': .3, 'bootstrap_draws': 2000, 'finite_draws': 2000,
                          'clusters': 43 if cohort == 'test' else 45,
                          'comparison_role': 'historical_reference' if tool in renderer.HISTORICAL else 'primary_comparator'})
    summary, replicates, dispersion, macro, paired = [], [], [], [], []
    for tool in renderer.SHOWN:
        seeds = [-1] if tool == 'guide_thermodynamic_ridge' else [0, 1, 2]
        for cohort, source in renderer.SOURCES:
            summary.append({'tool': tool, 'axis': 'grouped', 'cohort': cohort, 'stratum': 'source',
                            'stratum_value': source, 'n_min': 44, 'n_max': 44,
                            **{metric + '_mean': .2 for metric in ['pearson_r', 'r2', 'mae']}})
        for cell in ['h1299', 'halacat', 'hek293', 'hek293t', 'hep3b', 't24']:
            summary.append({'tool': tool, 'axis': 'grouped', 'cohort': 'test', 'stratum': 'cell_line',
                            'stratum_value': cell, 'n_min': 44, 'n_max': 44,
                            **{metric + '_mean': .2 for metric in ['pearson_r', 'r2', 'mae']}})
        for axis, folds in [('grouped', [-1]), ('random', range(5))]:
            for fold, seed in itertools.product(folds, seeds):
                replicates.append({'tool': tool, 'axis': axis, 'evaluation_fold': fold, 'training_seed': seed,
                                   'cohort': 'test', 'stratum': 'pooled', 'pearson_r': .2 + seed / 1000,
                                   'r2': .1 + seed / 1000})
            for metric in ['pearson_r', 'r2']:
                dispersion.append({'tool': tool, 'axis': axis, 'cohort': 'test', 'stratum': 'pooled',
                                   'metric': metric, 'mean': .2, 'sd': .01,
                                   'variation': 'training_seed_oof' if axis == 'grouped' else 'fold_of_seed_means'})
        for cohort, stratum, count in [('test', 'source', 7), ('test', 'cell_line', 6), ('hela_full', 'source', 4)]:
            for seed in seeds:
                macro.append({'tool': tool, 'axis': 'grouped', 'cohort': cohort, 'stratum': stratum,
                              **{metric + '_mean': .2 for metric in ['pearson_r', 'r2', 'mae']},
                              **{metric + '_count': count for metric in ['pearson_r', 'r2', 'mae']}})
    for (left, right), cohort, metric in itertools.product(itertools.combinations(renderer.NAMES, 2),
                                                          ['test', 'hela_full'], ['pearson_r', 'r2']):
        paired.append({'left': left, 'right': right, 'cohort': cohort, 'metric': metric,
                       'estimate_left_minus_right': .02, 'ci_lower_left_minus_right': -.01,
                       'ci_upper_left_minus_right': .05})
    frames = {'intervals': pd.DataFrame(intervals), 'summary': pd.DataFrame(summary), 'ranks': ranks(),
              'replicates': pd.DataFrame(replicates), 'dispersion': pd.DataFrame(dispersion),
              'macro': pd.DataFrame(macro), 'paired': pd.DataFrame(paired)}
    for name, filename in renderer.FILES.items():
        frames[name].to_csv(directory / filename, index=False)
    (directory / 'analysis_manifest.json').write_text(json.dumps(
        {'methods': list(renderer.NAMES), 'bootstrap_draws': 2000, 'bootstrap_seed': 20260921}))


def test_full_synthetic_render_writes_separate_artifacts_and_refuses_overwrite(tmp_path, monkeypatch):
    analysis, figures, tex_path = tmp_path / 'analysis', tmp_path / 'figures', tmp_path / 'preview' / 'expanded.tex'
    synthetic_analysis(analysis)
    monkeypatch.setattr(sys, 'argv', ['renderer', '--analysis', str(analysis), '--figures', str(figures),
                                     '--tex-output', str(tex_path)])
    rendered = []
    close = renderer.plt.close
    monkeypatch.setattr(renderer.plt, 'close', lambda figure: rendered.append(figure)
                        if isinstance(figure, renderer.plt.Figure) else None)
    renderer.main()
    assert len(rendered) == 3
    for figure in rendered:
        figure.canvas.draw()
        canvas = figure.canvas.get_renderer()
        for footer in figure.texts:
            box = footer.get_window_extent(canvas)
            for axis in figure.axes:
                for label in [*axis.get_xticklabels(), *axis.get_yticklabels(), axis.xaxis.label, axis.yaxis.label]:
                    if label.get_visible() and label.get_text():
                        assert not box.overlaps(label.get_window_extent(canvas))
        for axis in figure.axes:
            xlow, xhigh = sorted(axis.get_xlim())
            ylow, yhigh = sorted(axis.get_ylim())
            for collection in axis.collections:
                if isinstance(collection, PathCollection):
                    points = collection.get_offsets()
                    assert np.all((points[:, 0] >= xlow) & (points[:, 0] <= xhigh))
                    assert np.all((points[:, 1] >= ylow) & (points[:, 1] <= yhigh))
        close(figure)
    text = tex_path.read_text()
    assert '30 TabPFN gradient fine-tuning runs and 30 frozen TabPFN context fits' in text
    assert 'eight internal inference ensemble members' in text
    assert 'TabPFN tuned--frozen' in text
    assert '../figures/evaluation_robustness.pdf' in text
    for stem in ['evaluation_robustness', 'source_performance', 'rank_variation']:
        for extension in ['png', 'pdf', 'svg']:
            assert (figures / f'{stem}.{extension}').stat().st_size > 1000
    manifest = json.loads((figures / 'manifest.json').read_text())
    assert manifest['eligible_methods'] == 13 and len(manifest['displayed_methods']) == 9
    assert all(renderer.sha256(path) == digest for path, digest in manifest['outputs'].items())
    with pytest.raises(ValueError, match='existing supplement artifacts are preserved'):
        renderer.main()
