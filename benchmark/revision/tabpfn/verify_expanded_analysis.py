"""Check that adding TabPFN preserves all original numerical results."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_validation import sha256, write_json

NEW_TOOLS = {'tabpfn35_frozen', 'tabpfn35_finetuned'}
TABLES = ['replicate_metrics.csv', 'metric_summary.csv', 'replicate_dispersion.csv',
          'macro_metrics.csv', 'primary_group_bootstrap.csv', 'primary_paired_differences.csv']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--expanded', type=Path, required=True)
    parser.add_argument('--tabpfn-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the existing analysis-verification report')
    old_manifest = json.loads((args.original/'analysis_manifest.json').read_text())
    new_manifest = json.loads((args.expanded/'analysis_manifest.json').read_text())
    old_tools = set(old_manifest['methods'])
    if set(new_manifest['methods']) != old_tools | NEW_TOOLS:
        raise ValueError('Unexpected method set in expanded analysis')
    for key in old_manifest.keys() - {'methods', 'inputs'}:
        if old_manifest[key] != new_manifest[key]:
            raise ValueError(f'Changed numerical analysis setting: {key}')
    for path, digest in old_manifest['inputs'].items():
        if new_manifest['inputs'].get(path) != digest:
            raise ValueError(f'Original input changed: {path}')
    checked = {}
    for filename in TABLES:
        original = pd.read_csv(args.original/filename)
        expanded = pd.read_csv(args.expanded/filename)
        if 'tool' in expanded:
            preserved = expanded[expanded.tool.isin(old_tools)]
        else:
            preserved = expanded[expanded.left.isin(old_tools) & expanded.right.isin(old_tools)]
        pd.testing.assert_frame_equal(original.reset_index(drop=True), preserved.reset_index(drop=True), check_exact=True)
        checked[filename] = {'unchanged_rows': len(original), 'expanded_sha256': sha256(args.expanded/filename)}

    # Independent pooled R2 calculation from saved outer-test predictions.
    index = pd.read_csv(args.tabpfn_index)
    intervals = pd.read_csv(args.expanded/'primary_group_bootstrap.csv')
    independent_r2 = {}
    for tool in sorted(NEW_TOOLS):
        seed_metrics = []
        for seed in range(3):
            rows = index[(index.tool == tool) & (index.axis == 'grouped') & (index.training_seed == seed)]
            if len(rows) != 5:
                raise ValueError('Expected five grouped folds per seed')
            values = pd.concat([pd.read_csv(path) for path in rows.test_predictions], ignore_index=True)
            if len(values) != 3051 or not values.record_id.is_unique:
                raise ValueError('Incomplete/duplicated pooled outer-test predictions')
            residual_sum = np.square(values.label-values.pred_label).sum()
            total_sum = np.square(values.label-values.label.mean()).sum()
            seed_metrics.append(float(1-residual_sum/total_sum))
        observed = intervals[(intervals.tool == tool) & (intervals.cohort == 'test') & (intervals.metric == 'r2')]
        if len(observed) != 1 or not np.isclose(observed.estimate.iloc[0], np.mean(seed_metrics), rtol=0, atol=1e-12):
            raise ValueError('Independent pooled R2 differs from expanded analysis')
        independent_r2[tool] = {'seed_r2': seed_metrics, 'mean_r2': float(np.mean(seed_metrics))}
    write_json(args.output, {'status': 'verified', 'original_metrics_and_intervals_unchanged': checked,
                            'independent_grouped_r2': independent_r2,
                            'ranking_note': 'Ranks intentionally change when legitimate methods are added.',
                            'code_sha256': sha256(Path(__file__))})
    print('Original numerical results unchanged; both TabPFN pooled R2 estimates independently verified.')


if __name__ == '__main__':
    main()
