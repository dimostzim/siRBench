import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='16'
import argparse, json, random, time, warnings, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, scipy, joblib
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from helpers.training_reporter import TrainingReporter
from augmented_representation import encode, AugmentedNormalizer
from folding_features import SETTINGS

def load(path):
    path=Path(path)
    df=pd.read_csv(path/'input/data.csv',dtype={'id':str},keep_default_na=False)
    lab=pd.read_csv(path/'labels.csv',dtype={'id':str},keep_default_na=False)
    assert df.id.is_unique and lab.id.is_unique and set(df.id)==set(lab.id) and all(df.id.str.len()>0)
    col='numeric_label' if 'numeric_label' in lab else 'label'
    y=df[['id']].merge(lab[['id',col]],on='id',validate='one_to_one',sort=False)[col].to_numpy(dtype=float)
    assert np.isfinite(y).all() and ((y>=0)&(y<=1)).all()
    for g,c in zip(df.siRNA,df.extended_mRNA):
        assert len(g)==19 and set(g)<=set('ACGU') and len(c)==57 and set(c)<=set('ACGTX')
        assert g.replace('U','T').translate(str.maketrans('ACGT','TGCA'))[::-1]==c[19:38]
    return df,y,{str(f.relative_to(path)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (path/'input/data.csv',path/'labels.csv')}

def metrics(y,p):
    return dict(R2=float(r2_score(y,p)),Pearson=float(np.corrcoef(y,p)[0,1]),RMSE=float(np.sqrt(mean_squared_error(y,p))),MAE=float(mean_absolute_error(y,p)))

def main():
    parser=argparse.ArgumentParser()
    for name in ('train-data','validation-data','artifacts-dir'): parser.add_argument('--'+name,required=True)
    a=parser.parse_args(); out=Path(a.artifacts_dir);out.mkdir(parents=True,exist_ok=True)
    logs=out.parent/'training_logs';logs.mkdir(exist_ok=True)
    seed=int(os.environ.get('AGENTOMICS_TRAIN_SEED','0'));random.seed(seed);np.random.seed(seed)
    versions={k:m.__version__ for k,m in [('numpy',np),('pandas',pd),('scikit-learn',sklearn),('scipy',scipy),('joblib',joblib)]}
    versions['ViennaRNA']='2.7.2'
    policy=dict(iteration=17,total_search_iterations=20,exploration_iterations=3,candidates=4,seed=seed,threads=16,versions=versions,stopping='libsvm convergence tol=1e-4, max_iter=-1',grid=dict(structure_weight=[0.,.05,.2,.8]),folding=SETTINGS,determinism='SVR has no random_state; Python and NumPy use requested base seed')
    (logs/'policy.json').write_text(json.dumps(policy,indent=2))
    reporter=TrainingReporter();reporter.report_unavailable('sklearn LIBSVM exposes no epoch or batch callback; sequential fixed four-configuration search to convergence.')
    tr,y,ht=load(a.train_data);va,v,hv=load(a.validation_data)
    # Framework smoke validation may intentionally pass the same split twice.
    same_split=Path(a.train_data).resolve()==Path(a.validation_data).resolve()
    if not same_split:
        assert not set(tr.id)&set(va.id)
    else:
        warnings.warn('Same split supplied for smoke testing; metrics are not development validation evidence.')
    (logs/'audit.json').write_text(json.dumps(dict(train_rows=len(tr),validation_rows=len(va),train_hashes=ht,validation_hashes=hv,integrity_passed=True),indent=2))
    raw=encode(tr); valraw=encode(va)
    settings=dict(C=.2,gamma=.1,epsilon=.02,kernel='rbf',shrinking=True,tol=1e-4,cache_size=1024,max_iter=-1)
    rows=[];best=None;predictions=va[['id']].copy()
    for w in policy['grid']['structure_weight']:
        config=dict(structure_weight=w)
        start=time.time();row=dict(config=config,seed=seed,full_settings=settings)
        try:
            norm=AugmentedNormalizer(**config).fit(raw)
            x=norm.transform(raw);z=norm.transform(valraw)
            row['effective_weights']={b['name']:b['weight'] for b in norm.base.state['blocks']}
            model=SVR(**settings)
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always');model.fit(x,y)
            p=model.predict(z)
            row.update(train=metrics(y,model.predict(x)),validation=metrics(v,p),support_vectors=len(model.support_),fit_status=int(model.fit_status_),warnings=[str(w.message) for w in ws])
            predictions[f'candidate_{len(rows):02d}']=p
            key=(row['validation']['R2'],-w)
            if best is None or key>best[0]:best=(key,config,p.copy())
        except Exception as e:row['failure']=repr(e)
        row['elapsed_seconds']=time.time()-start;rows.append(row)
        (logs/'candidates.json').write_text(json.dumps(rows,indent=2))
    assert best is not None
    assert all('failure' not in r for r in rows), 'See candidates.json for failures'
    norm=AugmentedNormalizer(**best[1]).fit(raw);x=norm.transform(raw);z=norm.transform(valraw)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');model=SVR(**settings).fit(x,y)
    p=model.predict(z);assert np.allclose(p,best[2],rtol=0,atol=1e-12)
    joblib.dump(model,out/'model.joblib');norm.save(out)
    (out/'feature_schema.json').write_text((Path(__file__).parent/'feature_schema.json').read_text())
    control=next(r['validation']['R2'] for r in rows if r['config']==dict(structure_weight=0.))
    selection=dict(**policy,selected=best[1],svr=settings,training_rows=len(tr),validation=metrics(v,p),constituent_predictors=1,selection_reason='Maximum global validation R2; exact ties smaller structure weight',refit_reproduced=True,refit_status=int(model.fit_status_),refit_warnings=[str(w.message) for w in ws],control_R2=control,control_reference=0.608712,control_difference=control-0.608712)
    (out/'augmented_feature_schema.json').write_text((Path(__file__).parent/'augmented_feature_schema.json').read_text())
    (out/'config.json').write_text(json.dumps(selection,indent=2))
    predictions['prediction']=p;predictions.to_csv(logs/'validation_predictions.csv',index=False)
    (logs/'selection.json').write_text(json.dumps(selection,indent=2))
    (out.parent/'requirements.txt').write_text('\n'.join(f'{k}=={val}' for k,val in versions.items())+'\n')
    reporter.report_epoch(epoch=1,train_loss=mean_squared_error(y,model.predict(x)),val_loss=mean_squared_error(v,p),val_metric_name='R2',val_metric=r2_score(v,p))
    print(json.dumps(dict(selected=best[1],validation=metrics(v,p))))
if __name__=='__main__':main()
