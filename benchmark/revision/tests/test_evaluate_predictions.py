import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / 'evaluate_predictions.py'
sys.path.insert(0,str(PATH.parent))
spec = importlib.util.spec_from_file_location('evaluation',PATH)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_metrics_handle_perfect_and_constant_predictions():
    truth = [.1,.3,.8]
    scores = evaluation.metrics(truth,truth)
    np.testing.assert_allclose(list(scores.values()),[1,1,1,0,0,0])
    constant = evaluation.metrics(truth,[.4,.4,.4])
    assert np.isnan(constant['pearson_r'])
    assert np.isfinite(constant['r2'])
    assert np.isnan(evaluation.metrics([.1],[.2])['r2'])


def test_group_bootstrap_keeps_identical_methods_paired():
    truth = np.array([.1,.2,.8,.9])
    predictions = np.array([truth+.2,truth+.1])
    distributions = evaluation.group_bootstrap(truth,{'a':predictions,'b':predictions.copy()},np.array(['x','x','y','y']),repetitions=30,seed=4)
    np.testing.assert_allclose(distributions['a'],distributions['b'])
    np.testing.assert_allclose(distributions['a'][:,3], .025)


def fixture():
    records = pd.DataFrame({'record_id':['t','h'],'efficiency':[.2,.8],'cell_line':['a','HeLa']})
    membership = pd.DataFrame([{'axis':axis,'fold':fold,'part':'test','record_id':'t'} for axis in ['grouped','random'] for fold in range(5)])
    rows = []
    for axis in ['grouped','random']:
        for fold in range(5):
            for cohort,record,label in [('test','t',.2),('hela_full','h',.8)]:
                rows.append({'tool':'fixed','axis':axis,'fold':fold,'training_seed':-1,'cohort':cohort,'record_id':record,'label':label,'pred_label':.5})
    return pd.DataFrame(rows),records,membership


def test_validator_accepts_reordering_but_rejects_wrong_labels_and_missing_runs():
    predictions,records,membership = fixture()
    evaluation.validate_predictions(predictions.sample(frac=1,random_state=5),records,membership)
    wrong = predictions.copy()
    wrong.loc[0,'label'] = .8
    with pytest.raises(ValueError,match='labels differ'):
        evaluation.validate_predictions(wrong,records,membership)
    with pytest.raises(ValueError,match='incomplete evaluation matrix'):
        evaluation.validate_predictions(predictions.iloc[1:],records,membership)


def test_validator_rejects_duplicate_prediction_ids():
    predictions,records,membership = fixture()
    with pytest.raises(ValueError,match='Duplicate prediction'):
        evaluation.validate_predictions(pd.concat([predictions,predictions.iloc[:1]]),records,membership)


def test_historical_training_overlap_is_excluded_from_primary_ranks():
    rows = []
    for tool in ['guide_ridge','iscore_2007_fixed','iscore_2007_calibrated']:
        for identifier,label in [('a',.1),('b',.8)]:
            rows.append({'tool':tool,'axis':'grouped','fold':0,'training_seed':-1,'cohort':'test','record_id':identifier,'label':label,'pred_label':label})
    ranks = evaluation.fold_rankings(pd.DataFrame(rows)).set_index('tool')
    assert ranks.loc['guide_ridge','r2_rank'] == 1
    assert ranks.loc['iscore_2007_fixed','comparison_role'] == 'historical_reference'
    assert np.isnan(ranks.loc['iscore_2007_calibrated','r2_rank'])


def test_incomplete_group_metadata_cannot_silently_drop_predictions():
    predictions,records,_ = fixture()
    records['source'] = 'source'
    records['hela_aligned'] = [False,True]
    groups = pd.DataFrame({'record_id':['t'],'target_group_id':['g'],'split_group_id':['g']})
    with pytest.raises(ValueError,match='exactly the same unique IDs'):
        evaluation.add_metadata(predictions,records,groups)


def test_grouped_metrics_pool_oof_rows_before_averaging_seed_metrics():
    rows = []
    for seed, offset in [(0, -.1), (1, .1)]:
        for fold, truth in [(0, [.1,.2]), (1, [.8,.9])]:
            for j, label in enumerate(truth):
                rows.append({'tool':'model','axis':'grouped','fold':fold,
                    'training_seed':seed,'cohort':'test','record_id':f'{fold}_{j}',
                    'label':label,'pred_label':label+offset,'source':'source',
                    'cell_line':'cell','target_group_id':f'g{fold}'})
    result = evaluation.per_run_metrics(pd.DataFrame(rows))
    pooled = result[result.stratum.eq('pooled')]
    assert pooled.n.tolist() == [4,4]
    assert pooled.evaluation_fold.tolist() == [-1,-1]
    np.testing.assert_allclose(pooled.r2, .92)
    np.testing.assert_allclose(pooled.mse.mean(), .01)
    # Averaging the two prediction vectors would yield zero error instead.
    assert pooled.mse.mean() > 0


def test_primary_bootstrap_estimates_average_metrics_without_prediction_ensembling():
    rows = []
    identifiers = ['a','b','c','d']
    truth = [.1,.2,.8,.9]
    for cohort in ['test','hela_full','hela_aligned']:
        for seed, offset in [(0,-.1),(1,.1)]:
            for j,(record,label) in enumerate(zip(identifiers,truth)):
                folds = [j//2] if cohort == 'test' else [0,1]
                for fold in folds:
                    rows.append({'tool':'model','axis':'grouped','fold':fold,
                        'training_seed':seed,'cohort':cohort,'record_id':record,
                        'label':label,'pred_label':label+offset})
    records = pd.DataFrame({'record_id':identifiers,'efficiency':truth})
    groups = pd.DataFrame({'record_id':identifiers,'target_group_id':['g1','g1','g2','g2'],
                           'split_group_id':['g1','g1','g2','g2']})
    intervals,_ = evaluation.bootstrap_primary(pd.DataFrame(rows),records,groups,20)
    np.testing.assert_allclose(intervals[intervals.metric.eq('mse')].estimate,.01)
    np.testing.assert_allclose(intervals[intervals.metric.eq('r2')].estimate,.92)



def test_read_index_checks_seals_and_seed_identity(tmp_path):
    prediction = 'id,label,pred_label\na,0.1,0.2\n'
    paths = {}
    for column in ['test_predictions','hela_predictions','train_meta']:
        path = tmp_path / column
        path.write_text(json.dumps({'seed':0}) if column == 'train_meta' else prediction)
        paths[column] = str(path)
    row = {'tool':'model','axis':'grouped','fold':0,'training_seed':0,'status':'verified',**paths}
    index = tmp_path/'index.csv'
    pd.DataFrame([row]).to_csv(index,index=False)
    assert len(evaluation.read_run_index(index)) == 2
    row['test_predictions_sha256'] = '0'*64
    pd.DataFrame([row]).to_csv(index,index=False)
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        evaluation.read_run_index(index)
    row.pop('test_predictions_sha256')
    row['training_seed'] = 1
    pd.DataFrame([row]).to_csv(index,index=False)
    with pytest.raises(ValueError,match='Training seed differs'):
        evaluation.read_run_index(index)


def test_artifact_paths_cannot_be_reused_across_supplied_indexes(tmp_path):
    paths = {}
    for column in ['test_predictions','hela_predictions','train_meta']:
        path = tmp_path / column
        path.write_text(json.dumps({'seed':0}) if column == 'train_meta' else 'id,label,pred_label\na,0.1,0.2\n')
        paths[column] = str(path)
    indexes = []
    for tool in ['model_a','model_b']:
        path = tmp_path / (tool+'.csv')
        pd.DataFrame([{'tool':tool,'axis':'grouped','fold':0,'training_seed':0,'status':'verified',**paths}]).to_csv(path,index=False)
        indexes.append(path)
    with pytest.raises(ValueError,match='reused across run identities'):
        evaluation.read_run_indexes(indexes)


def dispersion_fixture():
    rows = []
    for fold, means in [(0,[1.,2.,3.]),(1,[5.,6.,7.])]:
        for seed,value in enumerate(means):
            rows.append({'tool':'model','comparison_role':'primary_comparator','axis':'random',
                         'cohort':'test','stratum':'pooled','stratum_value':'all',
                         'evaluation_fold':fold,'training_seed':seed,'n':10,'target_groups':2,
                         **{metric:value for metric in evaluation.METRICS}})
    return pd.DataFrame(rows)


def test_dispersion_separates_seed_sd_and_sd_of_fold_means():
    result = evaluation.replicate_dispersion(dispersion_fixture())
    result = result[result.metric.eq('mse')]
    within = result[result.variation.eq('training_seed_within_fold')]
    between = result[result.variation.eq('fold_of_seed_means')]
    np.testing.assert_allclose(within['mean'],[2,6])
    np.testing.assert_allclose(within.sd,[1,1])
    np.testing.assert_allclose(between.sd,[np.sqrt(8)])
    assert between.total_replicates.tolist() == [2]


def test_undefined_seed_is_preserved_in_summary_and_dispersion():
    runs = dispersion_fixture()
    runs.loc[0,'pearson_r'] = np.nan
    summary = evaluation.summarize_replicates(runs).iloc[0]
    assert np.isnan(summary.pearson_r_mean)
    assert np.isnan(summary.pearson_r_std)
    assert summary.pearson_r_count == 5
    assert summary.replicate_count == 6
    dispersion = evaluation.replicate_dispersion(runs)
    between = dispersion[dispersion.metric.eq('pearson_r') & dispersion.variation.eq('fold_of_seed_means')].iloc[0]
    assert np.isnan(between['mean'])
    assert between.finite_replicates == 1
    assert between.total_replicates == 2


def test_oof_seed_dispersion_has_no_artificial_between_fold_sd():
    runs = dispersion_fixture().iloc[:3].copy()
    runs['axis'] = 'grouped'
    runs['evaluation_fold'] = -1
    result = evaluation.replicate_dispersion(runs)
    assert set(result.variation) == {'training_seed_oof'}
    np.testing.assert_allclose(result.sd,1.)


def test_deterministic_fit_has_no_within_fold_seed_sd():
    runs = dispersion_fixture().query('training_seed == 0').copy()
    runs['training_seed'] = -1
    result = evaluation.replicate_dispersion(runs)
    within = result[result.variation.eq('training_seed_within_fold')]
    assert within.sd.isna().all()
    assert within.total_replicates.eq(1).all()
