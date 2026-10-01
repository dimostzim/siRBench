"""Aligned metrics, descriptive replication and paired group-bootstrap uncertainty."""
import argparse
import hashlib
import itertools
from io import BytesIO
from importlib.metadata import version
import platform
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from prediction_artifacts import read_artifact_bytes, validate_artifact_paths

METRICS = ('pearson_r', 'spearman_rho', 'r2', 'mse', 'mae', 'rmse')


def comparison_role(tool):
    return 'historical_reference' if tool.startswith('iscore_2007') else 'primary_comparator'


def correlation(truth, prediction):
    if len(truth) < 2 or np.ptp(truth) == 0 or np.ptp(prediction) == 0:
        return float('nan')
    return float(np.corrcoef(truth, prediction)[0,1])


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth,float), np.asarray(prediction,float)
    if truth.shape != prediction.shape or not len(truth) or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('Metrics require aligned, nonempty, finite vectors')
    squared_error = np.sum((truth-prediction)**2)
    total = np.sum((truth-truth.mean())**2)
    return {'pearson_r':correlation(truth,prediction),
            'spearman_rho':correlation(rankdata(truth),rankdata(prediction)),
            'r2':float(1-squared_error/total) if len(truth)>1 and total>0 else float('nan'),
            'mse':float(squared_error/len(truth)),
            'mae':float(np.mean(np.abs(truth-prediction))),
            'rmse':float(np.sqrt(squared_error/len(truth)))}


def read_run_index(path):
    return read_run_indexes([path])


def read_run_indexes(paths):
    rows = []
    index = pd.concat([pd.read_csv(path) for path in paths],ignore_index=True)
    if not index.status.eq('verified').all():
        raise ValueError('Unverified runs in prediction indexes')
    validate_artifact_paths(index)
    for row in index.itertuples(index=False):
        read_artifact_bytes(row, 'train_meta')
        for cohort, column in [('test','test_predictions'), ('hela_full','hela_predictions')]:
            payload, _ = read_artifact_bytes(row, column)
            frame = pd.read_csv(BytesIO(payload)).rename(columns={'id':'record_id'})
            frame = frame[['record_id','label','pred_label']].copy()
            frame['tool'],frame['axis'],frame['fold'] = row.tool,row.axis,row.fold
            frame['training_seed'],frame['cohort'] = row.training_seed,cohort
            rows.append(frame)
    return pd.concat(rows,ignore_index=True)


def validate_predictions(predictions, records, membership):
    key = ['tool','axis','fold','training_seed','cohort','record_id']
    if predictions.duplicated(key).any():
        raise ValueError('Duplicate prediction identity')
    truth = records.set_index('record_id').efficiency
    if not predictions.record_id.isin(truth.index).all():
        raise ValueError('Unknown prediction record ID')
    if not np.allclose(predictions.label, truth.loc[predictions.record_id], rtol=0,atol=1e-12):
        raise ValueError('Prediction labels differ from frozen records')
    if not np.isfinite(predictions[['label','pred_label']].to_numpy()).all():
        raise ValueError('Nonfinite prediction or label')
    expected_tests = {key:set(frame.record_id) for key,frame in membership[membership.part.eq('test')].groupby(['axis','fold'])}
    hela = set(records.loc[records.cell_line.str.lower().eq('hela'),'record_id'])
    for key,frame in predictions.groupby(key[:-1]):
        _,axis,fold,_,cohort = key
        expected = expected_tests[(axis,fold)] if cohort == 'test' else hela
        if cohort not in {'test','hela_full'} or set(frame.record_id) != expected:
            raise ValueError(f'Incomplete or unexpected prediction cohort {key}')
    for tool,frame in predictions.groupby('tool'):
        seeds = {-1} if set(frame.training_seed) == {-1} else {0,1,2}
        actual = set(map(tuple, frame[['axis','fold','training_seed','cohort']].drop_duplicates().to_numpy()))
        expected = set(itertools.product(['grouped','random'], range(5), seeds, ['test','hela_full']))
        if actual != expected:
            raise ValueError(f'{tool}: incomplete evaluation matrix ({len(actual)}/{len(expected)} cohorts)')


def add_metadata(predictions, records, groups):
    if not records.record_id.is_unique or not groups.record_id.is_unique or set(groups.record_id) != set(records.record_id):
        raise ValueError('Record/group metadata must cover exactly the same unique IDs')
    if groups[['target_group_id','split_group_id']].isna().any().any():
        raise ValueError('Missing target/split group metadata')
    metadata = records[['record_id','source','cell_line','hela_aligned']].merge(groups,on='record_id',validate='one_to_one')
    merged = predictions.merge(metadata,on='record_id',validate='many_to_one')
    aligned = merged[merged.cohort.eq('hela_full') & merged.hela_aligned].copy()
    aligned['cohort'] = 'hela_aligned'
    return pd.concat([merged,aligned],ignore_index=True)


def per_run_metrics(predictions):
    rows = []
    # Grouped test predictions are pooled out of fold once per record and seed.
    frame = predictions.copy()
    frame['evaluation_fold'] = frame.fold
    frame.loc[frame.axis.eq('grouped') & frame.cohort.eq('test'), 'evaluation_fold'] = -1
    keys = ['tool','axis','evaluation_fold','training_seed','cohort']
    for key,block in frame.groupby(keys):
        strata = [('pooled','all',block)]
        for category in ['source','cell_line']:
            strata.extend((category,str(value),part) for value,part in block.groupby(category))
        for category,value,part in strata:
            rows.append({**dict(zip(keys,key)), 'stratum':category,'stratum_value':value,
                         'comparison_role':comparison_role(key[0]),
                         'n':len(part),'target_groups':part.target_group_id.nunique(),
                         **metrics(part.label,part.pred_label)})
    return pd.DataFrame(rows)



SUMMARY_KEYS = ['tool','comparison_role','axis','cohort','stratum','stratum_value']


def replicate_mean(values):
    return float(np.mean(values)) if np.isfinite(values).all() else float('nan')


def replicate_std(values):
    return float(np.std(values,ddof=1)) if len(values)>1 and np.isfinite(values).all() else float('nan')


def summarize_replicates(runs):
    aggregations = {}
    for metric in METRICS:
        aggregations[metric+'_mean'] = (metric,replicate_mean)
        aggregations[metric+'_std'] = (metric,replicate_std)
        aggregations[metric+'_count'] = (metric,lambda values:int(np.isfinite(values).sum()))
    aggregations.update(replicate_count=('n','size'), n_min=('n','min'), n_max=('n','max'),
                        target_groups_min=('target_groups','min'), target_groups_max=('target_groups','max'))
    return runs.groupby(SUMMARY_KEYS).agg(**aggregations).reset_index()


def replicate_dispersion(runs):
    """Separate descriptive seed dispersion from dispersion of fold means."""
    values = runs.melt(id_vars=SUMMARY_KEYS+['evaluation_fold','training_seed'],
                      value_vars=list(METRICS),var_name='metric',value_name='value')
    keys = SUMMARY_KEYS+['metric']
    folds = values.groupby(keys+['evaluation_fold']).agg(
        mean=('value',replicate_mean), sd=('value',replicate_std),
        finite_replicates=('value',lambda values:int(np.isfinite(values).sum())),
        total_replicates=('value','size')).reset_index()
    folds['variation'] = np.where(folds.evaluation_fold.eq(-1),
                                  'training_seed_oof','training_seed_within_fold')
    between = folds[folds.evaluation_fold.ne(-1)].groupby(keys).agg(
        mean=('mean',replicate_mean), sd=('mean',replicate_std),
        finite_replicates=('mean',lambda values:int(np.isfinite(values).sum())),
        total_replicates=('mean','size')).reset_index()
    between['evaluation_fold'] = pd.NA
    between['variation'] = 'fold_of_seed_means'
    return pd.concat([folds,between],ignore_index=True)


def fold_rankings(predictions):
    rows = []
    tests = predictions[predictions.cohort.eq('test')]
    for key,frame in tests.groupby(['tool','axis','fold','training_seed']):
        rows.append({**dict(zip(['tool','axis','fold','training_seed'],key)),
                     **metrics(frame.label,frame.pred_label)})
    frame = pd.DataFrame(rows)
    seeds = sorted(set(frame.training_seed)-{-1})
    if seeds:
        fixed = frame[frame.training_seed.eq(-1)]
        copies = [fixed.assign(training_seed=seed) for seed in seeds]
        frame = pd.concat([frame[frame.training_seed.ne(-1)],*copies],ignore_index=True)
    frame['comparison_role'] = frame.tool.map(comparison_role)
    primary = frame[frame.comparison_role.eq('primary_comparator')]
    for metric in METRICS:
        frame[metric+'_rank'] = primary.groupby(['axis','fold','training_seed'])[metric].rank(method='average',ascending=metric in {'mse','mae','rmse'})
    return frame


def group_bootstrap(truth, predictions, groups, repetitions=2000, seed=20260921):
    """Same group draws for all methods; metric is averaged across training replicates."""
    rng = np.random.default_rng(seed)
    labels = np.unique(groups)
    members = [np.flatnonzero(groups == label) for label in labels]
    distributions = {tool:np.full((repetitions,len(METRICS)),np.nan) for tool in predictions}
    for iteration in range(repetitions):
        sampled = np.concatenate([members[i] for i in rng.integers(len(labels),size=len(labels))])
        for tool,replicates in predictions.items():
            values = [list(metrics(truth[sampled],p[sampled]).values()) for p in replicates]
            # Undefined correlations remain undefined; do not silently discard a seed.
            distributions[tool][iteration] = np.mean(values,axis=0)
    return distributions


def bootstrap_primary(predictions, records, groups, repetitions):
    eligible = predictions[predictions.axis.eq('grouped')]
    intervals, paired = [], []
    for cohort in ['test','hela_full','hela_aligned']:
        cohort_frame = eligible[eligible.cohort.eq(cohort)]
        ids = sorted(cohort_frame.record_id.unique())
        truth = records.set_index('record_id').loc[ids,'efficiency'].to_numpy()
        group_column = 'split_group_id' if cohort == 'test' else 'target_group_id'
        cluster = groups.set_index('record_id').loc[ids,group_column].to_numpy()
        matrices = {}
        for tool,frame in cohort_frame.groupby('tool'):
            replicate_keys = ['training_seed'] if cohort == 'test' else ['fold','training_seed']
            values = []
            for _,replicate in frame.groupby(replicate_keys):
                if not replicate.record_id.is_unique or set(replicate.record_id) != set(ids):
                    raise ValueError('Bootstrap requires complete, aligned prediction replicates')
                values.append(replicate.set_index('record_id').loc[ids,'pred_label'].to_numpy())
            matrices[tool] = np.array(values)
        distributions = group_bootstrap(truth,matrices,cluster,repetitions)
        point_estimates = {}
        for tool,array in distributions.items():
            point = np.mean([list(metrics(truth,p).values()) for p in matrices[tool]],axis=0)
            point_estimates[tool] = point
            for column,metric in enumerate(METRICS):
                finite = array[:,column][np.isfinite(array[:,column])]
                bounds = np.percentile(finite,[2.5,97.5]) if len(finite) else [np.nan,np.nan]
                intervals.append({'tool':tool,'comparison_role':comparison_role(tool),'cohort':cohort,'metric':metric,'estimate':point[column],
                                  'ci_lower':bounds[0],'ci_upper':bounds[1],'finite_draws':len(finite),
                                  'bootstrap_draws':repetitions,'clusters':len(np.unique(cluster))})
        for left,right in itertools.combinations(distributions,2):
            differences = distributions[left]-distributions[right]
            for column,metric in enumerate(METRICS):
                finite = differences[:,column][np.isfinite(differences[:,column])]
                bounds = np.percentile(finite,[2.5,97.5]) if len(finite) else [np.nan,np.nan]
                paired.append({'left':left,'right':right,'left_role':comparison_role(left),'right_role':comparison_role(right),
                               'primary_comparison':comparison_role(left)==comparison_role(right)=='primary_comparator','cohort':cohort,'metric':metric,
                               'estimate_left_minus_right':point_estimates[left][column]-point_estimates[right][column],
                               'ci_lower_left_minus_right':bounds[0],'ci_upper_left_minus_right':bounds[1],
                               'finite_draws':len(finite),'bootstrap_draws':repetitions})
    return pd.DataFrame(intervals), pd.DataFrame(paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--baselines',type=Path)
    parser.add_argument('--run-index',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bootstrap-draws',type=int,default=2000)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new, empty evaluation output directory')
    records = pd.read_csv(args.records)
    groups = pd.read_csv(args.protocol/'groups.csv')
    membership = pd.read_csv(args.protocol/'membership.csv')
    frames = [read_run_indexes(args.run_index)] if args.run_index else []
    if args.baselines: frames.append(pd.read_csv(args.baselines))
    if not frames: raise ValueError('No predictions supplied')
    predictions = pd.concat(frames,ignore_index=True)
    validate_predictions(predictions,records,membership)
    predictions = add_metadata(predictions,records,groups)
    args.output.mkdir(parents=True,exist_ok=True)
    runs = per_run_metrics(predictions)
    runs.to_csv(args.output/'replicate_metrics.csv',index=False)
    summarize_replicates(runs).to_csv(args.output/'metric_summary.csv',index=False)
    replicate_dispersion(runs).to_csv(args.output/'replicate_dispersion.csv',index=False)
    ranks = fold_rankings(predictions)
    ranks.to_csv(args.output/"fold_rankings.csv",index=False)
    macro_keys = ['tool','comparison_role','axis','evaluation_fold','training_seed','cohort','stratum']
    macro = runs[runs.stratum.isin(['source','cell_line'])].groupby(macro_keys)[list(METRICS)].agg(['mean','count']).reset_index()
    macro.columns = ['_'.join(filter(None,col)) for col in macro.columns]
    macro.to_csv(args.output/'macro_metrics.csv',index=False)
    intervals,paired = bootstrap_primary(predictions,records,groups,args.bootstrap_draws)
    intervals.to_csv(args.output/'primary_group_bootstrap.csv',index=False)
    paired.to_csv(args.output/'primary_paired_differences.csv',index=False)
    inputs = [args.records,args.protocol/'manifest.json',args.protocol/'groups.csv',args.protocol/'membership.csv',args.protocol/'run_matrix.csv',*args.run_index]
    for index_path in args.run_index:
        index = pd.read_csv(index_path)
        inputs.extend(Path(filename) for column in ['test_predictions','hela_predictions','train_meta'] for filename in index[column])
    if args.baselines:
        inputs.extend([args.baselines,args.baselines.parent/'manifest.json',args.baselines.parent/'selection.json'])
    report = {'software_versions':{'python':platform.python_version(),**{name:version(name) for name in ['numpy','pandas','scipy','scikit-learn']}},'methods':sorted(predictions.tool.unique()),'bootstrap_draws':args.bootstrap_draws,
        'bootstrap_seed':20260921,'confidence_intervals':'Percentile95%, conditional on the fitted models and frozen partitions; simultaneous group draws across methods. Grouped test pools out-of-fold predictions per training seed, then averages metrics over seeds. HeLa averages metrics over fold/seed predictors; predictions are not ensembled.',
        'clustering':'Non-HeLa target-and-similar-guide components; HeLa conservative target groups.',
        'replication':'replicate_dispersion.csv reports within-fold SD across seeds and between-fold SD of seed-averaged metrics separately. Grouped OOF test metrics have only seed dispersion after pooling folds. These are descriptive SDs, not independent-sample standard errors. metric_summary.csv retains overall replicate SD, which mixes fold and seed variation for random/HeLa cohorts. Deterministic baselines have one fit per partition and undefined within-fold seed SD.',
        'historical_references':'Both iScore variants retain supervised training overlap with2361Huesken guides. Their metrics/intervals are labeled historical_reference and excluded from primary ranks/superiority summaries; affine recalibration does not remove overlap.',
        'macro_metrics':'Unweighted mean of finite per-source or per-cell-line metrics for each evaluation replicate; finite subgroup counts accompany each metric.',
        'rankings':'Ranks per fold/seed are descriptive. Deterministic baselines are repeated only as comparands for each stochastic seed, not as independent fits.',
        'paired_comparisons':'Exploratory unadjusted paired intervals, no multiplicity-adjusted superiority claims or p-values. Positive delta favors left for correlations/R2, negative for MSE/MAE/RMSE.',
        'undefined_metrics':'Correlation is undefined for constant truth/predictions; R2 is undefined for constant truth or n<2. Replicate means and SDs preserve any undefined constituent metric; finite and total replicate counts are reported. Bootstrap estimates follow the same policy; interval bounds use only finite draws with finite/total counts. Macro metrics alone explicitly average finite subgroups with their finite counts.',
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'artifact_validation_sha256':hashlib.sha256(Path(__file__).with_name('prediction_artifacts.py').read_bytes()).hexdigest(),
        'inputs':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}}
    (args.output/'analysis_manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
