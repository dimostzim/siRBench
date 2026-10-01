"""Render verified original-six plus frozen/fine-tuned TabPFN-3.5 results."""
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

ORIGINAL_TOOLS = {'attsioff', 'sirnabert', 'ensirna', 'gnn4sirna', 'oligoformer', 'sirnadiscovery'}
TABPFN_TOOLS = {'tabpfn35_frozen', 'tabpfn35_finetuned'}
ORIGINAL_INDEX_SHA = '0449546d1115a1fe9b0f2c9f77527dbbc5c3eef570bc095d28a3844ce700ab4b'
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
    'tabpfn35_frozen': 'TabPFN-3.5 (frozen)',
    'tabpfn35_finetuned': 'TabPFN-3.5 (fine-tuned)',
}
HISTORICAL = {'iscore_2007_fixed', 'iscore_2007_calibrated'}
METRICS = ['pearson_r', 'spearman_rho', 'r2', 'mse', 'mae', 'rmse']
COHORTS = ['test', 'hela_full', 'hela_aligned']
COHORT_LABELS = {'test': 'Target-grouped out-of-fold', 'hela_full': 'Full HeLa transfer'}
CAPTION = (
    'Benchmark comparison with frozen and fine-tuned TabPFN-3.5. For the six published '
    'siRNA predictors and each TabPFN variant, each seed pools the five grouped test '
    'folds (3,051 unique records), then metrics are averaged over three seeds. Full '
    'HeLa metrics use all 1,047 records and average metrics from fifteen fold/seed '
    'predictors. Frozen TabPFN seeds vary preprocessing and ensemble randomness; '
    'fine-tuned TabPFN and the six siRNA predictors include gradient training. '
    'Deterministic reference baselines have one fit per partition. TabPFN variants '
    'use the same 176 sequence/thermodynamic inputs as combined ridge and no source '
    'or cell-line predictors. Intervals are 95% percentile intervals from 2,000 '
    'paired group-bootstrap draws (43 non-HeLa components or 45 HeLa target groups), '
    'conditional on fitted models and frozen partitions. Predictions are not '
    'averaged across fold/seed fits; TabPFN retains its eight internal inference '
    'ensemble members. Method order is fixed, not a ranking. Historical iScore variants '
    'are reported separately because of supervised overlap. Paired comparisons '
    'are exploratory and unadjusted for multiplicity.'
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_index(frame, tools):
    key = ['tool', 'axis', 'fold', 'training_seed']
    expected = set(itertools.product(tools, ['grouped', 'random'], range(5), range(3)))
    if frame.duplicated(key).any() or set(map(tuple, frame[key].to_numpy())) != expected:
        raise ValueError('Incomplete or duplicate run matrix')
    if not frame.status.eq('verified').all():
        raise ValueError('Unverified run in index')


def validate_intervals(frame):
    key = ['tool', 'cohort', 'metric']
    expected = set(itertools.product(set(LABELS) | HISTORICAL, COHORTS, METRICS))
    if frame.duplicated(key).any() or set(map(tuple, frame[key].to_numpy())) != expected:
        raise ValueError('Expected all fifteen methods and every cohort/metric exactly once')
    for row in frame.itertuples(index=False):
        role = 'historical_reference' if row.tool in HISTORICAL else 'primary_comparator'
        if row.comparison_role != role:
            raise ValueError('Inconsistent historical-reference role')
        if row.clusters != (43 if row.cohort == 'test' else 45):
            raise ValueError('Unexpected bootstrap clusters')
        if row.bootstrap_draws != 2000 or not 0 <= row.finite_draws <= 2000:
            raise ValueError('Unexpected bootstrap draw count')
        bounds = np.array([row.ci_lower, row.ci_upper])
        if row.finite_draws and (not np.isfinite(bounds).all() or bounds[0] > bounds[1]):
            raise ValueError('Invalid confidence interval')
        if not row.finite_draws and np.isfinite(bounds).any():
            raise ValueError('Interval without finite bootstrap draws')


def plot_metrics(frame, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 9.4), sharey=True)
    positions = np.arange(len(LABELS))
    for panel, (axis, (metric, cohort)) in enumerate(zip(
            axes.flat, itertools.product(['pearson_r', 'r2'], ['test', 'hela_full']))):
        values = frame.loc[frame.cohort.eq(cohort) & frame.metric.eq(metric)].set_index('tool')
        for position, tool in enumerate(LABELS):
            row = values.loc[tool]
            color = '#c56b22' if tool in TABPFN_TOOLS else '#146c94' if tool in ORIGINAL_TOOLS else '#656565'
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
    figure.subplots_adjust(left=.25, right=.985, top=.96, bottom=.14, wspace=.19, hspace=.24)
    figure.text(.25, .04, 'Points: replicate-mean metrics; lines: 95% group-bootstrap intervals.\n'
                'Blue: siRNA predictors. Orange: TabPFN variants. Grey: reference baselines.\n'
                'Frozen TabPFN replicates vary preprocessing/ensembles; other predictor replicates include training.', fontsize=8)
    for extension in ['pdf', 'svg', 'png']:
        figure.savefig(output/f'primary_performance.{extension}', dpi=220, facecolor='white')
    plt.close(figure)


def interval_text(row):
    if not np.isfinite(row.estimate):
        return '--'
    if not np.isfinite([row.ci_lower, row.ci_upper]).all():
        return f'{row.estimate:.3f} [undefined]'
    return f'{row.estimate:.3f} [{row.ci_lower:.3f}, {row.ci_upper:.3f}]'


def write_tables(frame, output):
    selected = frame.loc[frame.tool.isin(LABELS)]
    selected.to_csv(output/'primary_metrics.csv', index=False)
    frame.loc[frame.tool.isin(HISTORICAL)].to_csv(output/'historical_reference_metrics.csv', index=False)
    for cohort in ['test', 'hela_full']:
        values = selected.loc[selected.cohort.eq(cohort)].set_index(['tool', 'metric'])
        latex = [r'\begin{tabular}{lrrrr}', r'\toprule',
                 r'Method & Pearson $r$ & Spearman $\rho$ & $R^2$ & MAE \\', r'\midrule']
        markdown = ['| Method | Pearson r | Spearman rho | R² | MAE |', '|---|---|---|---|---|']
        for tool, label in LABELS.items():
            cells = [interval_text(values.loc[(tool, metric)]) for metric in ['pearson_r','spearman_rho','r2','mae']]
            latex.append(label + ' & ' + ' & '.join(cells) + r' \\')
            markdown.append('| ' + ' | '.join([label, *cells]) + ' |')
        latex += [r'\bottomrule', r'\end{tabular}']
        (output/f'{cohort}_metrics.tex').write_text('\n'.join(latex)+'\n')
        (output/f'{cohort}_metrics.md').write_text('\n'.join(markdown)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--original-index', type=Path, required=True)
    parser.add_argument('--tabpfn-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new, empty figure directory')
    analysis_path = args.analysis/'analysis_manifest.json'
    analysis = json.loads(analysis_path.read_text())
    if set(analysis['methods']) != set(LABELS) | HISTORICAL:
        raise ValueError('Incomplete expanded analysis')
    if analysis['bootstrap_draws'] != 2000 or analysis['bootstrap_seed'] != 20260921:
        raise ValueError('Analysis changed the frozen bootstrap policy')
    if sha256(args.original_index) != ORIGINAL_INDEX_SHA:
        raise ValueError('The original six-method index changed')
    for path, tools in [(args.original_index, ORIGINAL_TOOLS), (args.tabpfn_index, TABPFN_TOOLS)]:
        if analysis['inputs'].get(str(path)) != sha256(path):
            raise ValueError('Analysis does not use the supplied sealed index')
        validate_index(pd.read_csv(path), tools)
    intervals_path = args.analysis/'primary_group_bootstrap.csv'
    frame = pd.read_csv(intervals_path)
    validate_intervals(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    plot_metrics(frame, args.output)
    write_tables(frame, args.output)
    (args.output/'caption.txt').write_text(CAPTION+'\n')
    (args.output/'alt_text.txt').write_text(
        'Four panels compare thirteen methods using Pearson correlation and R-squared '
        'on grouped out-of-fold predictions and full HeLa transfer. Points show mean '
        'metrics; horizontal lines show group-bootstrap intervals when defined. The '
        'two orange rows show frozen and fine-tuned TabPFN-3.5. Undefined correlations '
        'are labeled. Method order is fixed rather than ranked.\n')
    files = [path for path in args.output.iterdir() if path.is_file()]
    report = {'description': CAPTION,
              'inputs': {str(path): sha256(path) for path in [analysis_path, args.original_index, args.tabpfn_index, intervals_path]},
              'code_sha256': sha256(Path(__file__)),
              'outputs': {path.name: sha256(path) for path in files}}
    (args.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
