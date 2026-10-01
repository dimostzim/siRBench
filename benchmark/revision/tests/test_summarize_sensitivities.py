import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import summarize_sensitivities as sensitivity


def create_run(tmp_path, name, seed=0, offset=.1, tool='model'):
    directory = tmp_path/name
    directory.mkdir()
    (directory/'inputs').mkdir()
    contents = {'train':(['a','b'],[.1,.9]), 'val':(['c','d'],[.2,.8]),
                'test':(['e','f'],[.3,.7]), 'leftout':(['h1','h2'],[.15,.85])}
    manifest = {'settings':{'seed':seed,'tools':[tool]},'inputs':{}}
    for role,(identifiers,labels) in contents.items():
        path = directory/'inputs'/(role+'.csv')
        pd.DataFrame({'record_id':identifiers,'efficiency':labels}).to_csv(path,index=False)
        manifest['inputs'][role] = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory/'manifest.json').write_text(json.dumps(manifest))
    row = {'tool':tool,'axis':'grouped','fold':0,'training_seed':seed,'status':'verified','run_dir':str(directory)}
    for role,column in [('test','test_predictions'),('leftout','hela_predictions')]:
        identifiers,labels = contents[role]
        path = directory/(role+'_pred.csv')
        pd.DataFrame({'id':identifiers,'label':labels,'pred_label':np.array(labels)+offset}).to_csv(path,index=False)
        row[column] = str(path)
    metadata = directory/'train_meta.json'
    metadata.write_text(json.dumps({'seed':seed}))
    row['train_meta'] = str(metadata)
    return row


def loaded_pair(tmp_path):
    reader = sensitivity.ArtifactReader()
    return [sensitivity.load_run(create_run(tmp_path,name,offset=offset),reader)
            for name,offset in [('reference',.2),('variant',.1)]]


def test_paired_metrics_align_ids_and_report_actual_hela_exclusions(tmp_path):
    reader = sensitivity.ArtifactReader()
    reference = sensitivity.load_run(create_run(tmp_path,'reference',offset=.2),reader)
    variant_row = create_run(tmp_path,'variant',offset=.1)
    path = Path(variant_row['test_predictions'])
    pd.read_csv(path).iloc[::-1].to_csv(path,index=False)
    variant = sensitivity.load_run(variant_row,reader)
    rows = sensitivity.compare_runs(reference,variant,{'h1','h2','h3'})
    assert rows[1]['n'] == 2 and rows[1]['excluded_full_hela'] == 1
    np.testing.assert_allclose(rows[0]['delta_mae'],-.1)
    np.testing.assert_allclose(rows[0]['delta_mse'],-.03)


@pytest.mark.parametrize('fault',['seed','ids','labels','same_run','same_artifact'])
def test_pairing_rejects_mismatched_identities(tmp_path,fault):
    reference,variant = loaded_pair(tmp_path)
    if fault=='seed':variant['row']['training_seed'] = 1
    elif fault=='ids':variant['inputs']['train'] = variant['inputs']['train'].rename(index={'a':'wrong'})
    elif fault=='labels':variant['inputs']['val'].iloc[0] += .01
    elif fault=='same_run':variant['row']['run_dir'] = reference['row']['run_dir']
    else:variant['row']['test_predictions'] = reference['row']['test_predictions']
    with pytest.raises(ValueError,match='mismatch|reuse'):
        sensitivity.compare_runs(reference,variant,{'h1','h2'})


def test_loaded_predictions_must_match_archived_labels_and_metadata_seed(tmp_path):
    row = create_run(tmp_path,'run')
    row['training_seed'] = 1
    with pytest.raises(ValueError,match='Training seed differs'):
        sensitivity.load_run(row,sensitivity.ArtifactReader())
    row['training_seed'] = 0
    path = Path(row['test_predictions'])
    frame = pd.read_csv(path)
    frame.loc[0,'label'] += .01
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match='Label mismatch'):
        sensitivity.load_run(row,sensitivity.ArtifactReader())


def test_manifest_hash_and_prediction_seals_are_checked(tmp_path):
    row = create_run(tmp_path,'run')
    row['prediction_sha256'] = '0'*64
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        sensitivity.load_run(row,sensitivity.ArtifactReader())
    row.pop('prediction_sha256')
    row.pop('test_predictions_sha256')
    path = Path(row['run_dir'])/'inputs/train.csv'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='Archived input hash differs'):
        sensitivity.load_run(row,sensitivity.ArtifactReader())


def test_delta_sd_is_computed_from_within_seed_differences(tmp_path):
    rows=[]
    for seed,offset in enumerate([.2,.3,.4]):
        reader=sensitivity.ArtifactReader()
        a=sensitivity.load_run(create_run(tmp_path,f'ref{seed}',seed,offset),reader)
        b=sensitivity.load_run(create_run(tmp_path,f'var{seed}',seed,.1),reader)
        for result in sensitivity.compare_runs(a,b,{'h1','h2'}):
            rows.append({'comparison':'case','tool':'model','training_seed':seed,**result})
    summary=sensitivity.summarize_deltas(pd.DataFrame(rows),{'case':{'expected_seeds':3,'status':'complete'}})
    mae=summary[summary.metric.eq('mae')]
    np.testing.assert_allclose(mae.delta_mean,-.2)
    np.testing.assert_allclose(mae.delta_sd,.1)
    # Undefined seed correlations propagate instead of dropping that seed.
    frame=pd.DataFrame(rows)
    frame.loc[0,['reference_pearson_r','delta_pearson_r']]=np.nan
    summary=sensitivity.summarize_deltas(frame,{'case':{'expected_seeds':3,'status':'complete'}})
    pearson=summary[summary.metric.eq('pearson_r') & summary.cohort.eq('test')].iloc[0]
    assert np.isnan(pearson.delta_mean) and pearson.delta_finite_seeds==2


def test_absent_and_unverified_runs_stay_pending(tmp_path):
    selector={'index':'missing.csv','match':{'axis':'grouped','fold':0}}
    descriptor={'comparisons':[{'id':'pending','tool':'model','description':'pending fits','cohort_description':'matched',
                               'seeds':[0,1,2],'reference':selector,'variant':selector}]}
    per_seed,summary,status=sensitivity.analyze(descriptor,tmp_path,{'h1','h2'},sensitivity.ArtifactReader())
    assert per_seed.empty and summary.empty
    assert status.iloc[0].status=='pending' and status.iloc[0].paired_seeds==0
    frame=pd.DataFrame([{'tool':'model','training_seed':0,'axis':'grouped','fold':0,'status':'running'}])
    assert sensitivity.select_run(frame,selector,'model',0) is None


def test_duplicate_selected_runs_are_rejected():
    row={'tool':'model','training_seed':0,'axis':'grouped','fold':0,'status':'verified'}
    with pytest.raises(ValueError,match='Duplicate selected run'):
        sensitivity.select_run(pd.DataFrame([row,row]),{'match':{'axis':'grouped','fold':0}},'model',0)



def test_reusing_seed_zero_artifacts_cannot_create_a_second_seed(tmp_path):
    selectors = {}
    for side in ['reference','variant']:
        row = create_run(tmp_path,side,seed=0)
        path = tmp_path/(side+'.csv')
        pd.DataFrame([row,{**row,'training_seed':1}]).to_csv(path,index=False)
        selectors[side] = {'index':path.name,'match':{'axis':'grouped','fold':0}}
    descriptor = {'comparisons':[{'id':'case','tool':'model','description':'paired',
                                 'cohort_description':'matched','seeds':[0,1],**selectors}]}
    with pytest.raises(ValueError,match='Training seed differs'):
        sensitivity.analyze(descriptor,tmp_path,{'h1','h2'},sensitivity.ArtifactReader())


@pytest.mark.parametrize('invalid_seed', [False, True])
def test_central_ensirna_completions_are_validated_and_aggregated(tmp_path, invalid_seed):
    selectors = {}
    for side in ['reference', 'variant']:
        row = create_run(tmp_path, side, tool='ensirna')
        row['status'] = 'complete'
        if invalid_seed and side == 'variant':
            Path(row['train_meta']).write_text(json.dumps({'seed': 7}))
        path = tmp_path/(side+'.csv')
        pd.DataFrame([row]).to_csv(path, index=False)
        selectors[side] = {'index': path.name, 'match': {'axis': 'grouped', 'fold': 0}}
    descriptor = {'comparisons': [{'id': 'en', 'tool': 'ensirna', 'description': 'paired',
                                  'cohort_description': 'matched', 'seeds': [0], **selectors}]}
    if invalid_seed:
        with pytest.raises(ValueError, match='Training seed differs'):
            sensitivity.analyze(descriptor, tmp_path, {'h1', 'h2'}, sensitivity.ArtifactReader())
    else:
        per_seed, summary, status = sensitivity.analyze(
            descriptor, tmp_path, {'h1', 'h2'}, sensitivity.ArtifactReader())
        assert status.iloc[0].status == 'complete' and len(per_seed) == 2
        assert len(summary) == 12


def test_other_tools_complete_status_is_not_accepted():
    row = {'tool': 'model', 'training_seed': 0, 'axis': 'grouped', 'fold': 0, 'status': 'complete'}
    assert sensitivity.select_run(pd.DataFrame([row]), {'match': {'axis': 'grouped'}}, 'model', 0) is None
