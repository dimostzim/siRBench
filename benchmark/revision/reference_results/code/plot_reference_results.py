"""Simplified primary-comparison plot with accepted fold-specific reference models."""
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ORIGINAL_TOOLS = {'attsioff', 'sirnabert', 'ensirna', 'gnn4sirna', 'oligoformer', 'sirnadiscovery'}


TABPFN_TOOLS = {'tabpfn35_frozen', 'tabpfn35_finetuned'}


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
    'sirbench_reference': 'Agentomics reference models',
}


HISTORICAL = {'iscore_2007_fixed', 'iscore_2007_calibrated'}


def plot_metrics(frame, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 16,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 11.2), sharey='row')
    for panel, (axis, (cohort, metric)) in enumerate(zip(
            axes.flat, itertools.product(['test', 'hela_full'], ['pearson_r', 'r2']))):
        values = frame.loc[frame.cohort.eq(cohort) & frame.metric.eq(metric)].set_index('tool')
        for position, tool in enumerate(LABELS):
            row = values.loc[tool]
            color = '#7b3294' if tool == 'sirbench_reference' else '#c56b22' if tool in TABPFN_TOOLS else '#146c94' if tool in ORIGINAL_TOOLS else '#656565'
            if np.isfinite(row.estimate):
                if np.isfinite([row.ci_lower, row.ci_upper]).all():
                    axis.hlines(position, row.ci_lower, row.ci_upper, color=color, lw=1.8)
                axis.plot(row.estimate, position, 'o', color=color, ms=5)
            else:
                axis.text(.98, position, 'undefined', ha='right', va='center',
                          transform=axis.get_yaxis_transform(), color='#777777', fontsize=13)
        labels = list(LABELS.values())
        labels[-1] = 'Agentomics (pooled OOF)' if cohort == 'test' else 'Agentomics (mean)'
        axis.axvline(0, color='#bbbbbb', lw=.7, zorder=0)
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.set_ylim(len(labels)-.5, -.5)
        axis.set_xlabel('Pearson r' if metric == 'pearson_r' else r'$R^2$')
        title = 'Grouped evaluation' if cohort == 'test' else 'HeLa transfer'
        axis.set_title(f'{chr(65+panel)}  {title}', loc='left', fontsize=17, pad=10)
        axis.spines[['top', 'right', 'left']].set_visible(False)
        axis.tick_params(axis='y', length=0)
        axis.grid(axis='x', color='#eeeeee', lw=.6, zorder=0)
        axis.margins(x=.08)
    legend = [Line2D([], [], marker='o', linestyle='', color=color, label=label)
              for color, label in [('#146c94', 'siRNA predictors'),
                                   ('#c56b22', 'TabPFN'),
                                   ('#656565', 'Baselines'),
                                   ('#7b3294', 'Agentomics')]]
    figure.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, 1.0),
                  ncol=4, frameon=False, fontsize=15, columnspacing=1.2,
                  handletextpad=.35)
    figure.subplots_adjust(left=.355, right=.985, top=.925, bottom=.065, wspace=.24, hspace=.26)
    for extension in ['pdf', 'svg', 'png']:
        figure.savefig(output/f'primary_performance.{extension}', dpi=300, facecolor='white')
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
