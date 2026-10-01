"""Plot cohort composition after freezing membership; do not select observations."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose an empty output directory')
    records = pd.read_csv(args.records)
    summary = pd.read_csv(args.summary).set_index('cohort')
    if not records.record_id.is_unique or not records.efficiency.between(0, 1).all():
        raise ValueError('Invalid record identifiers or efficacy values')
    is_hela = records.cell_line.str.lower().eq('hela')
    cohorts = {'nonhela': records[~is_hela], 'hela_full': records[is_hela],
               'hela_aligned': records[is_hela & records.hela_aligned]}
    colors = {'nonhela': '#146c94', 'hela_full': '#c76528', 'hela_aligned': '#8463a6'}
    names = {'nonhela': 'Non-HeLa', 'hela_full': 'Full HeLa', 'hela_aligned': 'Aligned HeLa (secondary)'}
    descriptions = []
    for name, frame in cohorts.items():
        statistic = ks_2samp(cohorts['nonhela'].efficiency, frame.efficiency).statistic
        if len(frame) != summary.loc[name, 'n'] or not np.isclose(statistic, summary.loc[name, 'ks_statistic_against_nonhela']):
            raise ValueError('Cohort disagrees with frozen descriptive audit')
        descriptions.append({'cohort': name, 'n': len(frame), 'ks_against_nonhela': statistic})
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    figure, (distribution, composition) = plt.subplots(1, 2, figsize=(10, 4.3),
                                                     gridspec_kw={'width_ratios': [1.2, 1]})
    for name, frame in cohorts.items():
        values = np.sort(frame.efficiency.to_numpy())
        distribution.step(values, np.arange(1, len(values)+1)/len(values), where='post',
                          label=f'{names[name]} (n={len(values):,})', color=colors[name],
                          linestyle='--' if name == 'hela_aligned' else '-', lw=1.6)
    distribution.set(xlim=(0, 1), ylim=(0, 1), xlabel='Efficacy', ylabel='Cumulative fraction')
    distribution.set_title('A  Outcome distributions after cohort selection', loc='left', fontsize=10)
    distribution.legend(loc='upper left', frameon=False, fontsize=8)
    statistics = {row['cohort']: row['ks_against_nonhela'] for row in descriptions}
    distribution.text(.98, .05, f"KS against non-HeLa\nFull HeLa: {statistics['hela_full']:.3f}\n"
                      f"Aligned HeLa: {statistics['hela_aligned']:.3f}",
                      ha='right', va='bottom', transform=distribution.transAxes, fontsize=8)
    counts = records.assign(cohort=np.where(is_hela, 'hela_full', 'nonhela')).groupby(
        ['source', 'cohort']).size().unstack(fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    y = np.arange(len(counts))
    for name, offset in [('nonhela', -.18), ('hela_full', .18)]:
        composition.barh(y+offset, counts[name], height=.34, color=colors[name], label=names[name])
        for position, value in zip(y+offset, counts[name]):
            if value: composition.text(value+25, position, str(value), va='center', fontsize=7)
    display_names = {'Simone': 'Sciabola (Simone)', 'ui-tei': 'Ui-Tei'}
    composition.set_yticks(y, [display_names.get(name, name.capitalize()) for name in counts.index])
    composition.set_ylim(len(counts)-.65, -.65)
    composition.set_xlim(0, counts.max().max()*1.16)
    composition.set_xlabel('Records')
    composition.set_title('B  Released source annotations', loc='left', fontsize=10)
    composition.legend(frameon=False, loc='lower right', fontsize=8)
    for axis in [distribution, composition]:
        axis.spines[['top', 'right']].set_visible(False)
    figure.subplots_adjust(left=.065, right=.975, top=.90, bottom=.15, wspace=.40)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'svg', 'png']:
        figure.savefig(args.output/f'cohort_distributions.{extension}', dpi=220, facecolor='white')
    plt.close(figure)
    counts.to_csv(args.output/'source_counts.csv')
    pd.DataFrame(descriptions).to_csv(args.output/'cohort_statistics.csv', index=False)
    caption = ('Outcome distributions and source composition of the revised legacy-data benchmark. '
        'A: empirical cumulative efficacy distributions for 3,051 non-HeLa records, the full 1,047-row HeLa pool '
        'and the historical aligned 896-row HeLa subset. KS statistics are descriptive comparisons after '
        'membership is frozen; they are not used to select revised data or partitions. '
        'B: counts by released source annotation. Huesken/H1299 dominates non-HeLa and Takayuki dominates '
        'HeLa. Source labels retain documented provenance uncertainties; the panel does not resolve historical attribution.')
    (args.output/'caption.txt').write_text(caption+'\n')
    (args.output/'alt_text.txt').write_text('Cumulative efficacy curves show residual distribution differences '
        'between non-HeLa, full HeLa and aligned HeLa. Source-count bars show 2,361 Huesken records in non-HeLa '
        'and 702 Takayuki records in HeLa; other sources are substantially smaller.\n')
    files = [p for p in args.output.iterdir() if p.is_file()]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps({'purpose': __doc__,
        'inputs': {str(p): digest(p) for p in [args.records, args.summary, Path(__file__)]},
        'outputs': {p.name: digest(p) for p in files}}, indent=2)+'\n')


if __name__ == '__main__':
    main()
