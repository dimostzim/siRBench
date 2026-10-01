"""Render a separate supplement and response continuation from verified expanded results."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from render_benchmark_results import LABELS, HISTORICAL, validate_intervals

SIRNA = ['attsioff', 'sirnabert', 'ensirna', 'gnn4sirna', 'oligoformer', 'sirnadiscovery']
TABPFN = ['tabpfn35_frozen', 'tabpfn35_finetuned']
METHODS = SIRNA + TABPFN
SHOWN = ['guide_thermodynamic_ridge'] + METHODS
NAMES = {**LABELS, 'iscore_2007_fixed': 'i-Score, fixed',
         'iscore_2007_calibrated': 'i-Score, calibrated'}
ORDER = [tool for tool in NAMES if tool not in METHODS] + METHODS
METRICS = ['pearson_r', 'spearman_rho', 'r2', 'mae', 'mse', 'rmse']
METRIC_LABELS = [r'$r$', r'$\rho$', r'$R^2$', 'MAE', 'MSE', 'RMSE']
SOURCES = [('test', source) for source in
           ['Huesken', 'Simone', 'amarz', 'hsieh', 'khvorova', 'reynolds', 'vickers']]
SOURCES += [('hela_full', source) for source in ['Takayuki', 'Shabalina', 'harborth', 'ui-tei']]
SOURCE_NAMES = {'Simone': 'Sciabola', 'amarz': 'Amarz', 'hsieh': 'Hsieh',
                'khvorova': 'Khvorova', 'reynolds': 'Reynolds', 'vickers': 'Vickers',
                'harborth': 'Harborth', 'ui-tei': 'Ui-Tei'}
FILES = {'summary': 'metric_summary.csv', 'replicates': 'replicate_metrics.csv',
         'intervals': 'primary_group_bootstrap.csv', 'ranks': 'fold_rankings.csv',
         'dispersion': 'replicate_dispersion.csv', 'macro': 'macro_metrics.csv',
         'paired': 'primary_paired_differences.csv'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_one(frame, **criteria):
    selected = frame
    for column, value in criteria.items():
        selected = selected[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f'Expected one result for {criteria}, found {len(selected)}')
    return selected.iloc[0]


def validate_ranks(frame):
    keys = ['tool', 'axis', 'fold', 'training_seed']
    expected = set(itertools.product(set(NAMES), ['grouped', 'random'], range(5), range(3)))
    if frame.duplicated(keys).any() or set(map(tuple, frame[keys].to_numpy())) != expected:
        raise ValueError('Incomplete or duplicate expanded fold-ranking matrix')
    eligible = frame[frame.tool.isin(LABELS)]
    if not eligible.comparison_role.eq('primary_comparator').all():
        raise ValueError('Incorrect primary rank eligibility')
    historical = frame[frame.tool.isin(HISTORICAL)]
    if not historical.comparison_role.eq('historical_reference').all():
        raise ValueError('Incorrect historical rank eligibility')
    if eligible.loc[eligible.tool.eq('training_mean'), 'pearson_r'].notna().any():
        raise ValueError('Constant training-mean predictor has a finite within-fold correlation')
    for metric, count in [('pearson_r', 12), ('r2', 13)]:
        groups = eligible.groupby(['axis', 'fold', 'training_seed'])[metric]
        if not groups.agg(lambda values: np.isfinite(values).sum()).eq(count).all():
            raise ValueError(f'Expected {count} finite eligible {metric} values per fold/seed')
        if not np.allclose(groups.rank(ascending=False), eligible[metric + '_rank'], equal_nan=True):
            raise ValueError('Expanded ranks disagree with the underlying metrics')
        if historical[metric + '_rank'].notna().any():
            raise ValueError('Historical reference received a primary rank')


def paired_cell(frame, left, right, cohort, metric):
    selected = frame[(frame.cohort == cohort) & (frame.metric == metric) &
                     (((frame.left == left) & (frame.right == right)) |
                      ((frame.right == left) & (frame.left == right)))]
    if len(selected) != 1:
        raise ValueError('Missing or duplicate paired contrast')
    row = selected.iloc[0]
    sign = 1 if row.left == left else -1
    estimate = sign * row.estimate_left_minus_right
    lower, upper = sorted([sign * row.ci_lower_left_minus_right,
                           sign * row.ci_upper_left_minus_right])
    if not np.isfinite([estimate, lower, upper]).all():
        return '--'
    return f'{estimate:+.3f} [{lower:+.3f}, {upper:+.3f}]'


def num(value):
    return '--' if pd.isna(value) else f'{value:.3f}'


def sd_num(value):
    if pd.notna(value) and 0 < abs(value) < .001:
        mantissa, exponent = f'{value:.2e}'.split('e')
        return '$' + mantissa + r'\times 10^{' + str(int(exponent)) + '}$'
    return num(value)


def tex(text):
    return str(text).replace('_', r'\_').replace('%', r'\%')


def save_figure(figure, directory, stem):
    for extension in ['png', 'pdf', 'svg']:
        figure.savefig(directory / f'{stem}.{extension}', dpi=220, bbox_inches='tight')
    plt.close(figure)


def plot_robustness(data, directory):
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 9), sharey=True)
    for column, metric in enumerate(['pearson_r', 'r2']):
        for position, tool in enumerate(SHOWN):
            rows = data['replicates']
            rows = rows[(rows.tool == tool) & (rows.cohort == 'test') & (rows.stratum == 'pooled')]
            grouped = rows[rows.axis == 'grouped'][metric].to_numpy()
            random = rows[rows.axis == 'random'].groupby('evaluation_fold')[metric].agg(
                lambda values: np.mean(values.to_numpy())).to_numpy()
            if len(grouped) != (1 if tool == 'guide_thermodynamic_ridge' else 3) or len(random) != 5:
                raise ValueError('Incomplete grouped or random metric replicates')
            for offset, values, color in [(-.13, grouped, '#176b87'), (.13, random, '#be6a25')]:
                jitter = np.linspace(-.035, .035, len(values)) if len(values) > 1 else np.zeros(1)
                axes[0, column].scatter(values, position + offset + jitter, s=22, color=color, alpha=.75)
                axes[0, column].vlines(np.mean(values), position + offset - .08,
                                       position + offset + .08, color=color, lw=2)
            points = [select_one(data['intervals'], tool=tool, cohort=cohort, metric=metric).estimate
                      for cohort in ['hela_full', 'hela_aligned']]
            axes[1, column].plot(points, [position - .13, position + .13], color='#999999', lw=1)
            axes[1, column].scatter(points, [position - .13, position + .13],
                                    color=['#176b87', '#be6a25'], s=30)
        for row in [0, 1]:
            axis = axes[row, column]
            axis.set_yticks(range(len(SHOWN)), [NAMES[tool] for tool in SHOWN])
            axis.set_xlabel('Pearson r' if metric == 'pearson_r' else r'$R^2$')
            axis.grid(axis='x', alpha=.2)
            title = 'Grouped and random test' if row == 0 else 'Full and aligned HeLa'
            axis.set_title(f'{chr(65 + row * 2 + column)}  {title}', loc='left')
    axes[0, 0].set_ylim(len(SHOWN) - .5, -.5)
    legend = [Line2D([], [], marker='o', linestyle='', color=color, label=label)
              for color, label in [('#176b87', 'Grouped / full HeLa'),
                                   ('#be6a25', 'Random / aligned HeLa')]]
    figure.legend(handles=legend, loc='upper center', bbox_to_anchor=(.60, 1.0),
                  ncol=2, frameon=False, fontsize=13, columnspacing=1.4,
                  handletextpad=.35)
    figure.subplots_adjust(left=.24, right=.99, top=.91, bottom=.07, hspace=.28, wspace=.17)
    save_figure(figure, directory, 'evaluation_robustness')


def source_rows(summary, tools, metric):
    values, labels, counts = [], [], []
    for cohort, source in SOURCES:
        rows = [select_one(summary, tool=tool, axis='grouped', cohort=cohort,
                           stratum='source', stratum_value=source) for tool in tools]
        count = int(rows[0].n_min)
        if any(row.n_min != count or row.n_max != count for row in rows):
            raise ValueError('Source cohorts differ between methods')
        values.append([row[metric + '_mean'] for row in rows])
        labels.append(f"{'OOF' if cohort == 'test' else 'HeLa'} / {SOURCE_NAMES.get(source, source)}")
        counts.append(count)
    return np.asarray(values), labels, counts


def plot_sources(data, directory):
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 12.2), sharex=True)
    for axis, metric, title in zip(axes, ['pearson_r', 'r2'], ['A  Pearson r', 'B  R²']):
        values, labels, counts = source_rows(data['summary'], SHOWN, metric)
        bounds = [-np.inf, -1, 0, .2, .4, .6, np.inf]
        colors = ['#b44449', '#efd0ce', '#eef1f4', '#c8e1e5', '#81becb', '#2c7c93']
        cmap = ListedColormap(colors)
        cmap.set_bad('#eeeeee')
        axis.imshow(values, cmap=cmap, norm=BoundaryNorm(bounds, len(colors)), aspect='auto')
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                axis.text(column, row, '--' if not np.isfinite(value) else f'{value:.2f}',
                          ha='center', va='center', fontsize=11,
                          color='white' if value < -1 or value >= .6 else '#1d2530')
        axis.axhline(6.5, color='white', lw=4)
        axis.set_yticks(range(len(labels)), [f'{label} ({count:,})' for label, count in zip(labels, counts)])
        axis.set_title(title, loc='left')
    axes[-1].set_xticks(range(len(SHOWN)), [NAMES[tool] for tool in SHOWN], rotation=40, ha='right')
    figure.subplots_adjust(left=.24, right=.99, top=.97, bottom=.19, hspace=.14)
    figure.text(.24, .025, 'Means: three OOF seeds or 15 HeLa fits; ridge: one OOF estimate or five HeLa fits.\n'
                'Frozen TabPFN seeds vary preprocessing/ensemble construction. Parentheses: source records.\n'
                'Dark red: < −1; pale red: −1–0; blue bins: 0–0.2, 0.2–0.4, 0.4–0.6, ≥0.6.\n'
                'Values are not clipped; small groups and differing outcome variance limit comparisons.', fontsize=10)
    save_figure(figure, directory, 'source_performance')


def plot_ranks(data, directory):
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 6.8), sharey=True)
    for axis, metric, title in zip(axes, ['pearson_r_rank', 'r2_rank'], ['A  Pearson r rank', 'B  R² rank']):
        values = [data['ranks'][(data['ranks'].axis == 'grouped') &
                               (data['ranks'].tool == tool)][metric].to_numpy() for tool in SHOWN]
        axis.boxplot(values, orientation='horizontal', tick_labels=[NAMES[tool] for tool in SHOWN],
                     widths=.5, showfliers=False, medianprops={'color': '#176b87'})
        for position, value in enumerate(values, 1):
            axis.scatter(value, position + np.linspace(-.12, .12, len(value)), s=10, alpha=.55, color='#176b87')
        axis.set_xticks(range(1, 14))
        axis.set_xlim(.5, 13.5)
        axis.set_xlabel('Rank (1 is highest)')
        axis.set_title(title, loc='left')
        axis.grid(axis='x', alpha=.2)
    axes[0].set_ylim(len(SHOWN) + .5, .5)
    figure.subplots_adjust(left=.24, right=.99, top=.94, bottom=.27, wspace=.15)
    figure.text(.24, .035, '13 eligible methods; 12 finite within-fold Pearson ranks (training-mean r is undefined).\n'
                'Dots: 15 grouped fold/seed comparisons; deterministic ridge reused for each seed.\n'
                'Frozen TabPFN seeds vary preprocessing/ensembles. Correlated ranks are descriptive.\n'
                'Historical i-Score references are excluded. All nine displayed methods use the same rank pool.', fontsize=10)
    save_figure(figure, directory, 'rank_variation')


def write_source_markdown(summary, path):
    tools = ['guide_thermodynamic_ridge'] + TABPFN
    lines = ['# Table R1 continuation: TabPFN feature-matched comparison', '',
             'Retain the original seven-method panels A and B unchanged. Append these narrow panels C and D. '
             'Each TabPFN value is the mean of three grouped out-of-fold seed metrics or 15 grouped-fold/seed HeLa metrics. '
             'Ridge uses one pooled out-of-fold estimate or five HeLa metrics. Frozen seeds vary preprocessing and '
             'ensemble construction. No predictions are averaged across these fits; internal TabPFN inference uses '
             'eight ensemble members. Source counts and provenance qualifications are unchanged.', '']
    for metric, title in [('pearson_r', 'C. Pearson r'), ('r2', 'D. R²')]:
        values, labels, counts = source_rows(summary, tools, metric)
        lines += [f'**{title}**', '', '| Evaluation / source | n | ' + ' | '.join(NAMES[tool] for tool in tools) + ' |',
                  '|---|---:|---:|---:|---:|']
        lines += ['| ' + ' | '.join([label, f'{count:,}', *(num(value) for value in row)]) + ' |'
                  for label, count, row in zip(labels, counts, values)]
        lines.append('')
    path.write_text('\n'.join(lines))


def write_source_latex(summary, summary_path, directory):
    tools = ['guide_thermodynamic_ridge'] + TABPFN
    caption = ('Table R1 (continued). Source-specific performance of combined ridge and the two '
               'TabPFN-3.5 variants. Panels C and D supplement the original seven-method panels A and B. '
               'Values are mean metrics across three grouped out-of-fold seeds or 15 grouped-fold/seed '
               'HeLa fits; ridge uses one pooled out-of-fold estimate or five HeLa fits. Frozen seeds vary '
               'preprocessing and ensemble construction. Predictions are not averaged across these fits; '
               'TabPFN retains eight internal inference ensemble members. The three methods receive the '
               'same 176 guide and central-duplex features. Source annotations retain the provenance '
               'qualifications described in the response. Small sources and differing outcome variances '
               'limit comparisons; negative R-squared values are retained.')
    lines = [r'\par\medskip\noindent\begin{minipage}{\linewidth}',
             r'{\small ' + caption + r'\par}\medskip',
             r'\centering\small\setlength{\tabcolsep}{7pt}\renewcommand{\arraystretch}{1.16}']
    cells = []
    for metric, title in [('pearson_r', 'C. Pearson $r$'), ('r2', 'D. $R^2$')]:
        values, _, counts = source_rows(summary, tools, metric)
        lines += [r'\textbf{' + title + r'}\par\smallskip',
                  r'\begin{tabular}{@{}lrrrr@{}}\toprule',
                  r'Source & $n$ & \shortstack{Combined\\ridge} & \shortstack{TabPFN-3.5\\frozen} & \shortstack{TabPFN-3.5\\fine-tuned} \\ \midrule']
        previous_cohort = None
        for (cohort, source), row, count in zip(SOURCES, values, counts):
            if cohort != previous_cohort:
                if previous_cohort is not None:
                    lines.append(r'\addlinespace[4pt]')
                heading = 'Non-HeLa grouped out-of-fold' if cohort == 'test' else 'Full HeLa transfer'
                lines.append(r'\multicolumn{5}{@{}l}{\textit{' + heading + r'}} \\')
                previous_cohort = cohort
            label = SOURCE_NAMES.get(source, source)
            formatted = [num(value) for value in row]
            cells.append({'metric': metric, 'cohort': cohort, 'source': source,
                          'label': label, 'n': count, 'values': formatted})
            lines.append(' & '.join([label, f'{count:,}', *formatted]) + r' \\')
        lines.append(r'\bottomrule\end{tabular}\par\medskip')
    lines.append(r'\end{minipage}\par\medskip')
    asset = directory / 'Table_R1_continuation.tex'
    asset.write_text('\n'.join(lines) + '\n')
    wrapper = directory / 'Table_R1_continuation_standalone.tex'
    wrapper.write_text('\n'.join([
        r'\documentclass[10pt,a4paper]{article}', r'\usepackage[margin=21mm]{geometry}',
        r'\usepackage{fontspec,booktabs}',
        r'\setmainfont{texgyretermes-regular.otf}[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf,BoldItalicFont=texgyretermes-bolditalic.otf]',
        r'\setlength{\parindent}{0pt}\pagestyle{empty}', r'\begin{document}',
        r'\input{Table_R1_continuation.tex}', r'\end{document}', '']))
    manifest = {'comment_id': 'R2.m5', 'label': 'Table R1 (continued)', 'caption': caption,
                'source': str(summary_path), 'source_sha256': sha256(summary_path),
                'asset': asset.name, 'sha256': sha256(asset), 'methods': tools, 'rows': cells}
    (directory / 'response_source_continuation.json').write_text(json.dumps(manifest, indent=2) + '\n')


def figure_start(path):
    return r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{\detokenize{' + path + '}}'


def write_tables(data, figure_paths, tex_output):
    # Existing table structures are retained; only scope, variants and figure paths change.
    lines=[r'\section{Primary and complementary evaluation results}',r'\label{sec:primarysupp}',
           'The expanded comparison contains 180 main retraining runs for six siRNA-specific predictors, 30 TabPFN gradient fine-tuning runs and 30 frozen TabPFN context fits. The original 48 sensitivity fits remain separate. Primary figures use group-bootstrap intervals conditional on the frozen folds and fitted models. Full machine-readable tables retain all metrics, bootstrap intervals, paired contrasts, subgroup counts and individual replicates.',
           r'\subsection{Pooled performance}',
           'Grouped test metrics pool the 3,051 out-of-fold records separately for each seed, then average the three metrics. HeLa metrics average over 15 fold/seed predictors without averaging predictions across those fits. TabPFN retains eight internal inference ensemble members. Frozen seeds vary preprocessing and ensemble construction; fine-tuned seeds additionally vary optimization. Deterministic references are evaluated once per partition. Historical i-Score variants are shown for context only and are excluded from primary rankings because their published coefficients were fitted on overlapping Huesken data. A dash denotes an undefined metric.']
    for cohort,title in [('test','Target-grouped out-of-fold evaluation'),('hela_full','Full HeLa transfer'),('hela_aligned','Previously aligned HeLa subset')]:
        lines += [r'\begin{table}[htbp]\centering\small',r'\caption{'+title+r': pooled metrics.}\label{tab:pooled-'+cohort.replace('_','-')+'}',r'\begin{tabular}{lrrrrrr}\toprule', 'Method & '+' & '.join(METRIC_LABELS)+r' \\ \midrule']
        for tool in ORDER:
            rows=data['intervals'][(data['intervals'].tool==tool)&(data['intervals'].cohort==cohort)].set_index('metric')
            lines.append(tex(NAMES[tool])+(' $^{*}$' if tool.startswith('iscore') else '')+' & '+' & '.join(num(rows.loc[m,'estimate']) for m in METRICS)+r' \\')
        lines += [r'\bottomrule\end{tabular}',r'\par\smallskip\footnotesize $^{*}$Historical coefficient overlap; not eligible for primary ranking.',r'\end{table}']
    lines += [r'\clearpage\subsection{Paired group-bootstrap contrasts}',
              'The following contrasts compare each predictor with combined guide/thermodynamic ridge. Intervals use the same 2,000 resampled target groups for both methods. They are exploratory percentile intervals without multiplicity adjustment. Positive differences favor the named predictor. The fine-tuned-minus-frozen contrast is also reported.']
    paired=data['paired']
    lines += [r'\begin{longtable}{llrr}\caption{Paired differences from combined ridge, and fine-tuned minus frozen TabPFN, with 95\% group-bootstrap intervals.}\label{tab:paired}\\',r'\toprule Cohort & Predictor & $\Delta r$ [95\% CI] & $\Delta R^2$ [95\% CI] \\ \midrule\endfirsthead',r'\toprule Cohort & Predictor & $\Delta r$ [95\% CI] & $\Delta R^2$ [95\% CI] \\ \midrule\endhead']
    for cohort,label in [('test','Grouped OOF'),('hela_full','Full HeLa')]:
        for left,right,title in [(tool,'guide_thermodynamic_ridge',NAMES[tool]) for tool in METHODS]+[('tabpfn35_finetuned','tabpfn35_frozen','TabPFN tuned--frozen')]:
            cells=[paired_cell(paired,left,right,cohort,metric) for metric in ['pearson_r','r2']]
            lines.append(label+' & '+title+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule\end{longtable}',r'\subsection{Partition and training variability}',
              'Random-test results use five unselected partitions with 2,498 training, 278 validation and 275 test records each. Their standard deviation below is across five seed-averaged partition metrics, not across 15 independent samples. Grouped out-of-fold standard deviation is across three seeds. For frozen TabPFN this describes preprocessing/ensemble variation, not gradient-training variation. Grouped training sizes vary from 1,919 to 2,381; different test/training sizes preclude attributing the entire random/grouped gap to target overlap.']
    lines += [r'\begin{table}[htbp]\centering\small',r'\caption{Seed and partition variation in test metrics.}\label{tab:variation}',r'\begin{tabular}{lrrrr}\toprule',r'& \multicolumn{2}{c}{Grouped OOF (seed SD)} & \multicolumn{2}{c}{Random (partition SD)} \\',r'Method & $r$ & $R^2$ & $r$ & $R^2$ \\ \midrule']
    for tool in METHODS:
        cells=[]
        for axis,var in [('grouped','training_seed_oof'),('random','fold_of_seed_means')]:
            for metric in ['pearson_r','r2']:
                row=data['dispersion'][(data['dispersion'].tool==tool)&(data['dispersion'].axis==axis)&(data['dispersion'].cohort=='test')&(data['dispersion'].stratum=='pooled')&(data['dispersion'].metric==metric)&(data['dispersion'].variation==var)].iloc[0]
                cells.append(num(row['mean']) + r' $\pm$ ' + sd_num(row.sd))
        lines.append(NAMES[tool]+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{table}',figure_start(figure_paths['evaluation_robustness']),r'\caption{Unselected random and grouped test performance, and full versus aligned HeLa. Upper panels display three grouped out-of-fold seed metrics per stochastic predictor (one for deterministic ridge), and five random-partition means. Lower panels display means across 15 fold/seed predictors (five for ridge). Frozen TabPFN seeds vary preprocessing and ensemble construction. No averaging of predictions across fold/seed fits is performed; TabPFN retains eight internal inference ensemble members. No paired grouping-effect inference is made between different protocols. Alt text: Nine methods are shown under grouped and random test protocols and on full versus aligned HeLa, with each protocol labeled separately.}\label{fig:robustness}\end{figure}',figure_start(figure_paths['rank_variation']),r'\caption{Grouped fold/seed rank variation. Each dot is one of 15 fold/seed evaluations. The comparison includes 13 eligible methods, of which twelve have finite within-fold $r$ because the training-mean predictor is constant. The six siRNA-specific predictors, two TabPFN variants and combined ridge are displayed. Each deterministic ridge fold result is reused across the three seed comparisons. Boxes show medians and interquartile ranges. Correlated folds/seeds do not constitute 15 independent observations. Alt text: Predictor ranks vary across folds and seeds, with overlapping ranges.}\label{fig:ranks}\end{figure}',r'\clearpage\subsection{Source and cell-line performance}',
              'Source and cell-line summaries use the retained annotations; uncertain historical assignments remain flagged in the provenance audit. Source names in the figures are readable forms of the original identifiers (Simone denotes Sciabola). Macro values weight each finite subgroup equally and therefore answer a different question from pooled record-level metrics. Differing outcome variances can produce large negative subgroup $R^2$ values; the raw values are retained rather than clipped.']
    lines += [figure_start(figure_paths['source_performance']),r'\caption{Per-source correlation and $R^2$ for grouped out-of-fold and full HeLa evaluations. Counts are shown in parentheses, and numeric values are means across the applicable fitted replicates. Color bins distinguish negative values and increasing positive performance; unclipped numeric values remain printed. Alt text: Two panels show source-specific Pearson correlation and R-squared for nine methods, with source counts and numeric values retained, including negative values.}\label{fig:subgroups}\end{figure}']
    lines += [r'\begin{longtable}{lllrrr}\caption{Per-cell-line grouped out-of-fold performance.}\label{tab:cells}\\',r'\toprule Cell line & Method & $n$ & $r$ & $R^2$ & MAE \\ \midrule\endfirsthead',r'\toprule Cell line & Method & $n$ & $r$ & $R^2$ & MAE \\ \midrule\endhead']
    for cell in ['h1299','halacat','hek293','hek293t','hep3b','t24']:
        for tool in SHOWN:
            row=data['summary'][(data['summary'].tool==tool)&(data['summary'].axis=='grouped')&(data['summary'].cohort=='test')&(data['summary'].stratum=='cell_line')&(data['summary'].stratum_value==cell)].iloc[0]
            lines.append(f'{cell} & {NAMES[tool]} & {int(row.n_min)} & '+ ' & '.join(num(row[m+'_mean']) for m in ['pearson_r','r2','mae'])+r' \\')
    lines += [r'\bottomrule\end{longtable}',r'\begin{longtable}{lllrrr}\caption{Unweighted macro averages over finite annotated subgroups.}\label{tab:macro}\\',r'\toprule Cohort / grouping & Method & Groups & $r$ & $R^2$ & MAE \\ \midrule\endfirsthead',r'\toprule Cohort / grouping & Method & Groups & $r$ & $R^2$ & MAE \\ \midrule\endhead']
    for cohort,stratum,label in [('test','source','OOF / source'),('test','cell_line','OOF / cell line'),('hela_full','source','HeLa / source')]:
        for tool in SHOWN:
            rows=data['macro'][(data['macro'].tool==tool)&(data['macro'].axis=='grouped')&(data['macro'].cohort==cohort)&(data['macro'].stratum==stratum)]
            counts=[rows[m+'_count'].unique().tolist() for m in ['pearson_r','r2','mae']]
            assert all(len(c)==1 for c in counts) and len({c[0] for c in counts})==1
            lines.append(f'{label} & {NAMES[tool]} & {int(counts[0][0])} & '+' & '.join(num(rows[m+'_mean'].mean()) for m in ['pearson_r','r2','mae'])+r' \\')
    lines += [r'\bottomrule\end{longtable}']
    tex_output.write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--figures', type=Path, required=True)
    parser.add_argument('--tex-output', type=Path, required=True)
    args = parser.parse_args()
    if (args.figures.exists() and any(args.figures.iterdir())) or args.tex_output.exists():
        raise ValueError('Choose new output paths; existing supplement artifacts are preserved')
    manifest = json.loads((args.analysis / 'analysis_manifest.json').read_text())
    if set(manifest['methods']) != set(NAMES) or manifest['bootstrap_draws'] != 2000 or manifest['bootstrap_seed'] != 20260921:
        raise ValueError('Expected the complete expanded analysis with frozen bootstrap settings')
    data = {name: pd.read_csv(args.analysis / filename) for name, filename in FILES.items()}
    validate_intervals(data['intervals'])
    validate_ranks(data['ranks'])
    args.figures.mkdir(parents=True, exist_ok=True)
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    plot_robustness(data, args.figures)
    plot_sources(data, args.figures)
    plot_ranks(data, args.figures)
    figure_paths = {stem: os.path.relpath(args.figures / f'{stem}.pdf', args.tex_output.parent)
                    for stem in ['evaluation_robustness', 'rank_variation', 'source_performance']}
    write_tables(data, figure_paths, args.tex_output)
    write_source_markdown(data['summary'], args.figures / 'response_source_continuation.md')
    write_source_latex(data['summary'], args.analysis / FILES['summary'], args.figures)
    inputs = [args.analysis / name for name in FILES.values()] + [args.analysis / 'analysis_manifest.json']
    outputs = list(args.figures.iterdir()) + [args.tex_output]
    report = {'analysis_inputs': {str(path): sha256(path) for path in inputs},
              'code_sha256': sha256(__file__),
              'shared_renderer_sha256': sha256(Path(__file__).with_name('render_benchmark_results.py')),
              'outputs': {str(path): sha256(path) for path in outputs},
              'eligible_methods': 13, 'finite_within_fold_pearson_ranks': 12,
              'displayed_methods': SHOWN,
              'note': 'Separate rendering of verified expanded results; original artifacts and 48 sensitivity fits unchanged.'}
    (args.figures / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(args.figures)


if __name__ == '__main__':
    main()
