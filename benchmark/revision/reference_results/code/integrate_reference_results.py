"""Add the frozen fold-specific reference pipelines without altering prior results."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--selected', type=Path, required=True)
parser.add_argument('--helpers', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.helpers))
import evaluate_predictions as evaluation
import plot_reference_results as plotting

output = args.output
output.mkdir(parents=True, exist_ok=False)
records_path = args.root/'datasets/corrected-v1/records_features.csv'
protocol = args.root/'evaluation/protocol-v1'
old_analysis = args.root/'evaluation/primary-analysis-with-tabpfn-v1'
records = pd.read_csv(records_path)
groups = pd.read_csv(protocol/'groups.csv')
membership = pd.read_csv(protocol/'membership.csv')
paths = [args.selected/'grouped_oof_predictions.csv', args.selected/'hela_per_model_predictions.csv']
frames = []
for cohort, path in zip(['test', 'hela_full'], paths):
    frame = pd.read_csv(path).rename(columns={'id':'record_id', 'prediction':'pred_label'})
    frame['tool'] = 'sirbench_reference'
    frame['axis'] = 'grouped'
    frame['training_seed'] = -1  # Unreplicated selected fit; no common internal training seed.
    frame['cohort'] = cohort
    assert set(frame.fold) == set(range(5))
    assert not frame.duplicated(['fold','record_id']).any()
    for fold, part in frame.groupby('fold'):
        expected = set(membership.loc[(membership.axis == 'grouped') & (membership.fold == fold) & (membership.part == 'test'), 'record_id']) if cohort == 'test' else set(records.loc[records.cell_line.str.lower().eq('hela'), 'record_id'])
        assert set(part.record_id) == expected, (cohort, fold)
    assert np.allclose(frame.label, records.set_index('record_id').loc[frame.record_id, 'efficiency'], rtol=0, atol=1e-12)
    assert np.isfinite(frame[['label','pred_label']]).all().all()
    frames.append(frame)
assert frames[0].record_id.is_unique and len(frames[0]) == 3051
new = evaluation.add_metadata(pd.concat(frames, ignore_index=True), records, groups)
new.to_csv(output/'reference_predictions_with_metadata.csv', index=False)
runs = evaluation.per_run_metrics(new)
runs.to_csv(output/'reference_replicate_metrics.csv', index=False)
summary = evaluation.summarize_replicates(runs)
summary.to_csv(output/'reference_metric_summary.csv', index=False)
summary.loc[summary.stratum.isin(['source','cell_line'])].to_csv(output/'reference_subgroup_metrics.csv', index=False)

# Read and checksum-validate the existing indexed predictions; their values remain unchanged.
indexes = [args.root/'evaluation/primary-index-v1/run_index.csv', args.root/'evaluation/tabpfn-benchmark-v1/run_index.csv']
old = evaluation.read_run_indexes(indexes)
baselines = args.root/'evaluation/baselines-v1/predictions.csv'
old = pd.concat([old, pd.read_csv(baselines)], ignore_index=True)
evaluation.validate_predictions(old, records, membership)
comparands = {'guide_thermodynamic_ridge','ensirna','sirnadiscovery','tabpfn35_frozen','tabpfn35_finetuned'}
old = evaluation.add_metadata(old.loc[old.tool.isin(comparands) & old.axis.eq('grouped')], records, groups)
intervals, paired = evaluation.bootstrap_primary(pd.concat([old,new], ignore_index=True), records, groups, 2000)
prior = pd.read_csv(old_analysis/'primary_group_bootstrap.csv')
# Recomputed comparands must reproduce frozen bootstrap results before adding the new row.
check = intervals.loc[intervals.tool.isin(comparands)].merge(prior, on=['tool','cohort','metric'], suffixes=('_new','_old'), validate='one_to_one')
for column in ['estimate','ci_lower','ci_upper']:
    assert np.allclose(check[column+'_new'], check[column+'_old'], rtol=0, atol=1e-12, equal_nan=True), column
new_intervals = intervals.loc[intervals.tool.eq('sirbench_reference')]
combined = pd.concat([prior, new_intervals], ignore_index=True)
combined.to_csv(output/'primary_group_bootstrap.csv', index=False)
new_intervals.to_csv(output/'reference_group_bootstrap.csv', index=False)
paired.loc[paired.left.eq('sirbench_reference') | paired.right.eq('sirbench_reference')].to_csv(output/'reference_paired_differences.csv', index=False)
figure_dir = output/'figures'
figure_dir.mkdir()
plotting.plot_metrics(combined, figure_dir)
plotting.write_tables(combined, figure_dir)
caption = ('Panels A and B show Pearson correlation and R² for target-grouped out-of-fold evaluation (3,051 records); panels C and D show full HeLa transfer (1,047 records). Blue denotes the six published siRNA predictors, orange the two TabPFN-3.5 variants, grey the reference baselines, and purple the Agentomics reference models. For the published predictors and TabPFN variants, grouped predictions are pooled across five folds within each seed and metrics are averaged over three seeds; HeLa metrics are averaged over fifteen fold/seed fits. The Agentomics reference-model row uses one validation-selected pipeline per fold: a single pooled out-of-fold estimate and the mean of five individual HeLa metrics. The fold-specific pipelines can have different architectures; their predictions are not averaged into a cross-fold ensemble. Each search used the same planned 20-iteration budget including three exploration iterations; completed validated iterations were 19, 20, 18, 18 and 20. Points show these estimates and lines show 95% percentile intervals from 2,000 paired group-bootstrap draws over 43 non-HeLa components or 45 HeLa target groups, conditional on the fitted models and fixed partitions. These intervals do not quantify repeat-search or training-seed uncertainty for the reference models. No source or cell-line predictors enter the new reference models. Method order is fixed rather than ranked; small point differences do not establish superiority.')
(figure_dir/'caption.txt').write_text(caption+'\n')
(figure_dir/'alt_text.txt').write_text('Four panels compare fourteen methods using Pearson correlation and R-squared in grouped evaluation and full HeLa transfer. Horizontal lines show conditional group-bootstrap intervals. The single purple Agentomics row shows the pooled grouped result or the mean of the five full-HeLa metrics.\n')
source_paths = [*paths, records_path, protocol/'groups.csv', protocol/'membership.csv', old_analysis/'primary_group_bootstrap.csv', *indexes, baselines, Path(__file__), args.helpers/'evaluate_predictions.py', Path(plotting.__file__)]
manifest = {'name':'Accepted fold-specific siRBench reference models', 'bootstrap_draws':2000, 'bootstrap_seed':20260921, 'completed_iterations':[19,20,18,18,20], 'selected_iterations_zero_based':[16,17,15,17,13], 'reference_fits':5, 'random_split_runs':0, 'comparands':sorted(comparands), 'prior_interval_invariance':'PASS within 1e-12', 'limitations':'One selected fit per fold; no search/seed replication. Conditional CIs and exploratory unadjusted paired differences; no multiplicity-adjusted superiority claim.', 'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
(output/'analysis_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(new_intervals.loc[new_intervals.metric.isin(['pearson_r','r2'])].to_string(index=False))
