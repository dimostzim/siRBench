"""Render the complete, verified primary comparison without re-ranking methods."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from assemble_primary_index import TOOLS


LABELS = {
    'training_mean': 'Training mean',
    'guide_ridge': 'Guide ridge',
    'thermodynamic_ridge': 'Thermodynamic ridge',
    'guide_thermodynamic_ridge': 'Combined ridge',
    'uitei_sidirect_calibrated': 'Ui-Tei score (calibrated)',
    'attsioff': 'AttSiOff',
    'sirnabert': 'BERT-siRNA',
    'ensirna': 'ENsiRNA',
    'gnn4sirna': 'GNN4siRNA',
    'oligoformer': 'OligoFormer',
    'sirnadiscovery': 'siRNADiscovery',
}
HISTORICAL = {'iscore_2007_fixed', 'iscore_2007_calibrated'}
METRICS = ['pearson_r', 'spearman_rho', 'r2', 'mse', 'mae', 'rmse']
COHORTS = ['test', 'hela_full', 'hela_aligned']
COHORT_LABELS = {'test': 'Target-grouped out-of-fold', 'hela_full': 'Full HeLa transfer'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_intervals(frame):
    key = ['tool', 'cohort', 'metric']
    expected = set(itertools.product(set(LABELS) | HISTORICAL, COHORTS, METRICS))
    if frame.duplicated(key).any() or set(map(tuple, frame[key].to_numpy())) != expected:
        raise ValueError('Expected every metric/cohort for all thirteen methods exactly once')
    for row in frame.itertuples(index=False):
        expected_role = 'historical_reference' if row.tool in HISTORICAL else 'primary_comparator'
        if row.comparison_role != expected_role:
            raise ValueError('Historical overlap role is inconsistent')
        if row.clusters != (43 if row.cohort == 'test' else 45):
            raise ValueError('Unexpected uncertainty resampling groups')
        if row.bootstrap_draws != 2000 or not 0 <= row.finite_draws <= row.bootstrap_draws:
            raise ValueError('Unexpected bootstrap draw count')
        bounds = np.array([row.ci_lower, row.ci_upper])
        if row.finite_draws and (not np.isfinite(bounds).all() or bounds[0] > bounds[1]):
            raise ValueError('Invalid confidence interval')
        if not row.finite_draws and np.isfinite(bounds).any():
            raise ValueError('Interval reported without finite bootstrap draws')


def interval_text(row):
    if not np.isfinite(row.estimate):
        return '--'
    if not np.isfinite([row.ci_lower, row.ci_upper]).all():
        return f'{row.estimate:.3f} [undefined]'
    return f'{row.estimate:.3f} [{row.ci_lower:.3f}, {row.ci_upper:.3f}]'


def plot_metrics(frame, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 8.2), sharey=True)
    positions = np.arange(len(LABELS))
    for panel, (axis, (metric, cohort)) in enumerate(zip(
            axes.flat, itertools.product(['pearson_r', 'r2'], ['test', 'hela_full']))):
        values = frame.loc[frame.cohort.eq(cohort) & frame.metric.eq(metric)].set_index('tool')
        for position, tool in enumerate(LABELS):
            row = values.loc[tool]
            color = '#146c94' if tool in TOOLS else '#656565'
            if np.isfinite(row.estimate):
                if np.isfinite([row.ci_lower, row.ci_upper]).all():
                    axis.hlines(position, row.ci_lower, row.ci_upper, color=color, lw=1.4)
                axis.plot(row.estimate, position, 'o', color=color, ms=4)
            else:
                axis.text(.98, position, 'undefined', ha='right', va='center',
                          transform=axis.get_yaxis_transform(), color='#777777', fontsize=8)
        axis.axvline(0, color='#bbbbbb', lw=.7, zorder=0)
        axis.set_yticks(positions, list(LABELS.values()))
        axis.set_ylim(len(LABELS)-.5, -.5)
        axis.set_xlabel('Pearson r' if metric == 'pearson_r' else r'$R^2$')
        axis.set_title(f'{chr(65+panel)}  {COHORT_LABELS[cohort]}', loc='left', fontsize=10)
        axis.spines[['top', 'right', 'left']].set_visible(False)
        axis.tick_params(axis='y', length=0)
        axis.grid(axis='x', color='#eeeeee', lw=.6, zorder=0)
        axis.margins(x=.08)
    figure.subplots_adjust(left=.245, right=.985, top=.96, bottom=.14, wspace=.19, hspace=.24)
    figure.text(.245, .035, 'Points: mean metrics across fitted replicates; lines: 95% group-bootstrap intervals.\n'
                'Blue: retrained predictors. Grey: reference baselines. Undefined correlations are retained.', fontsize=8)
    for extension in ['pdf', 'svg', 'png']:
        figure.savefig(output/f'primary_performance.{extension}', dpi=220, facecolor='white')
    plt.close(figure)


def write_tables(frame, output):
    selected = frame.loc[frame.tool.isin(LABELS)].copy()
    selected.to_csv(output/'primary_metrics.csv', index=False)
    frame.loc[frame.tool.isin(HISTORICAL)].to_csv(output/'historical_reference_metrics.csv', index=False)
    for cohort in ['test', 'hela_full']:
        values = selected.loc[selected.cohort.eq(cohort)].set_index(['tool', 'metric'])
        lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
                 r'Method & Pearson $r$ & Spearman $\rho$ & $R^2$ & MAE \\', r'\midrule']
        for tool, label in LABELS.items():
            cells = [interval_text(values.loc[(tool, metric)]) for metric in ['pearson_r','spearman_rho','r2','mae']]
            lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}']
        (output/f'{cohort}_metrics.tex').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose an empty output directory')
    analysis_path = args.analysis/'analysis_manifest.json'
    index_path = args.index/'manifest.json'
    analysis = json.loads(analysis_path.read_text())
    index = json.loads(index_path.read_text())
    run_index = args.index/'run_index.csv'
    if index['runs'] != 180 or set(index['tools']) != TOOLS:
        raise ValueError('The six-method primary matrix is incomplete')
    if index['index_sha256'] != sha256(run_index) or analysis['inputs'].get(str(run_index)) != sha256(run_index):
        raise ValueError('Analysis does not use the sealed primary index')
    if set(analysis['methods']) != set(LABELS) | HISTORICAL:
        raise ValueError('The primary analysis is incomplete')
    intervals_path = args.analysis/'primary_group_bootstrap.csv'
    frame = pd.read_csv(intervals_path)
    validate_intervals(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    plot_metrics(frame, args.output)
    write_tables(frame, args.output)
    description = ('Primary comparison on corrected legacy public data. For the six retrained predictors, '
        'each seed pools predictions across the five grouped test folds (3,051 records) before computing metrics; '
        'these metrics are then averaged across three seeds. Full HeLa metrics use all 1,047 records and '
        'are averaged across fifteen fold/seed fits. Deterministic baselines have one fit per partition, '
        'giving one pooled out-of-fold metric and the mean of five HeLa metrics. '
        'Intervals use 2,000 paired group-bootstrap draws (43 non-HeLa components or 45 HeLa target groups), '
        'conditional on fitted models and frozen partitions. Predictions are not ensembled. '
        'Historical i-Score references are reported separately because of supervised overlap. '
        'Method order is fixed and is not a ranking. No multiplicity-adjusted superiority claim is made.')
    (args.output/'caption.txt').write_text(description+'\n')
    (args.output/'alt_text.txt').write_text('Four panels compare eleven methods using Pearson correlation and R-squared '
        'on grouped out-of-fold predictions and the full HeLa transfer set. Each method has a point estimate '
        'and a horizontal confidence interval when defined. The training-mean baseline has undefined correlation '
        'on HeLa; its grouped out-of-fold predictions combine fold-specific training means.\n')
    files = [p for p in args.output.iterdir() if p.is_file()]
    report = {'description': description, 'inputs': {str(p): sha256(p) for p in [analysis_path,index_path,run_index,intervals_path]},
              'code_sha256': sha256(Path(__file__)), 'outputs': {p.name: sha256(p) for p in files}}
    (args.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
