"""Plot verified source-specific R² estimates for the manuscript."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

METHODS = {
    'guide_thermodynamic_ridge': 'Combined ridge',
    'attsioff': 'AttSiOff',
    'sirnabert': 'BERT-siRNA',
    'ensirna': 'ENsiRNA',
    'gnn4sirna': 'GNN4siRNA',
    'oligoformer': 'OligoFormer',
    'sirnadiscovery': 'siRNADiscovery',
    'tabpfn35_frozen': 'TabPFN (frozen)',
    'tabpfn35_finetuned': 'TabPFN (fine-tuned)',
    'sirbench_reference': 'Agentomics',
}
SOURCES = {
    'test': [('Huesken', 'Huesken'), ('Simone', 'Sciabola'), ('amarz', 'Amarzguioui'),
             ('hsieh', 'Hsieh'), ('khvorova', 'Khvorova'), ('reynolds', 'Reynolds'), ('vickers', 'Vickers')],
    'hela_full': [('Takayuki', 'Takayuki'), ('Shabalina', 'Shabalina'),
                  ('harborth', 'Harborth'), ('ui-tei', 'Ui-Tei')],
}


def plot_sources(summary, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    colors = ['#b44449', '#efd0ce', '#eef1f4', '#c8e1e5', '#81becb', '#2c7c93']
    intervals = ['< −1', '−1 to 0', '0 to 0.2', '0.2 to 0.4', '0.4 to 0.6', '≥ 0.6']
    bounds = [-np.inf, -1, 0, .2, .4, .6, np.inf]
    figure, axes = plt.subplots(2, 1, figsize=(8.8, 10.4))
    plotted = []
    for axis, (cohort, sources), panel in zip(axes, SOURCES.items(), ['A', 'B']):
        values = np.empty((len(METHODS), len(sources)))
        labels = []
        for column, (source, label) in enumerate(sources):
            selected = summary.loc[summary.axis.eq('grouped') & summary.cohort.eq(cohort) &
                                   summary.stratum.eq('source') & summary.stratum_value.eq(source) &
                                   summary.tool.isin(METHODS)].set_index('tool')
            if len(selected) != len(METHODS) or not selected.index.is_unique:
                raise ValueError(f'Incomplete source comparison: {cohort}/{source}')
            if selected.n_min.nunique() != 1 or not selected.n_min.eq(selected.n_max).all():
                raise ValueError('Source record counts differ between methods')
            count = int(selected.n_min.iloc[0])
            labels.append(f'{label}\n{count:,}')
            for row, tool in enumerate(METHODS):
                value = float(selected.loc[tool, 'r2_mean'])
                if not np.isfinite(value):
                    raise ValueError('A displayed source metric is undefined')
                values[row, column] = value
                plotted.append({'cohort': cohort, 'source': source, 'tool': tool, 'n': count, 'r2': value})
        axis.imshow(values, cmap=ListedColormap(colors), norm=BoundaryNorm(bounds, len(colors)), aspect='auto')
        for row, column in np.ndindex(values.shape):
            value = values[row, column]
            label = f'{value:.3f}' if 0 < abs(value) < .005 else f'{value:.2f}'
            axis.text(column, row, label, ha='center', va='center', fontsize=13,
                      color='white' if value < -1 or value >= .6 else '#1d2530')
        axis.set_yticks(range(len(METHODS)), list(METHODS.values()))
        axis.set_xticks(range(len(sources)), labels, fontsize=11)
        axis.tick_params(length=0, pad=6)
        axis.set_title(f'{panel}  '+('Grouped out-of-fold' if cohort == 'test' else 'Full HeLa transfer'),
                       loc='left', fontsize=16, pad=10)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.legend(handles=[Patch(facecolor=c, label=t) for c, t in zip(colors, intervals)],
                  loc='upper center', bbox_to_anchor=(.51, .995), ncol=6,
                  frameon=False, title=r'$R^2$', fontsize=11, columnspacing=.8,
                  handlelength=1.1, handletextpad=.4)
    figure.subplots_adjust(left=.27, right=.99, top=.885, bottom=.065, hspace=.29)
    for extension in ['pdf', 'png', 'svg']:
        figure.savefig(output/f'source_r2.{extension}', dpi=300, facecolor='white')
    plt.close(figure)
    pd.DataFrame(plotted).to_csv(output/'source_r2_plot_values.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--reference-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = pd.concat([pd.read_csv(args.summary), pd.read_csv(args.reference_summary)], ignore_index=True)
    plot_sources(summary, args.output)
    paths = [args.summary, args.reference_summary, Path(__file__)]
    (args.output/'source_r2_manifest.json').write_text(json.dumps({
        'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'cells': 110, 'models': list(METHODS),
        'aggregation': 'Existing grouped seed means and full-HeLa fit means; one grouped estimate and five HeLa fits for Agentomics.',
        'new_training_or_metrics': False,
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
