import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='16'
import argparse,json,random,time,platform,gc
from pathlib import Path
import numpy as np
import catboost,sklearn,joblib
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score,mean_squared_error
from threadpoolctl import threadpool_limits
from features import load_input,load_labels
from helpers.training_reporter import TrainingReporter

class Progress:
    def __init__(self,r): self.r=r; self.best=-float('inf'); self.bi=0
    def after_iteration(self,info):
        s=info.metrics['validation']['R2'][-1]
        if s>self.best: self.best=s; self.bi=info.iteration
        if info.iteration==1 or info.iteration%100==0:
            self.r.report_epoch(epoch=info.iteration,val_metric_name='R2',val_metric=s,early_stopping_patience_remaining=max(0,150-info.iteration+self.bi))
        return True

def main():
    ap=argparse.ArgumentParser()
    for n in ('train-data','validation-data','artifacts-dir'): ap.add_argument('--'+n,required=True,type=Path)
    a=ap.parse_args(); a.artifacts_dir.mkdir(parents=True,exist_ok=True)
    seed=int(os.environ.get('AGENTOMICS_TRAIN_SEED','0')); reporter=TrainingReporter()
    log={'budget':'Iteration 13 of 20 including three exploration iterations; fixed 4 CatBoost and 6 ExtraTrees candidates, two reproducibility refits; no adaptive expansion','seed':seed,'dependencies':dict(python=platform.python_version(),numpy=np.__version__,catboost=catboost.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__),'candidates':[]}
    lp=a.artifacts_dir.parent/(a.artifacts_dir.name+'_training_log.json')
    def save(): lp.write_text(json.dumps(log,indent=2,allow_nan=False))
    save()
    ids,X=load_input(a.train_data/'input'); y=load_labels(a.train_data/'labels.csv',ids)
    vi,V=load_input(a.validation_data/'input'); vy=load_labels(a.validation_data/'labels.csv',vi)
    assert X.shape[1]==V.shape[1]==801
    XF=np.ascontiguousarray(X,dtype=np.float32); VF=np.ascontiguousarray(V,dtype=np.float32)
    log.update(train_rows=len(y),validation_rows=len(vy),validation_ids=vi)
    cb=dict(loss_function='RMSE',eval_metric='R2',boosting_type='Ordered',grow_policy='SymmetricTree',learning_rate=.03,iterations=3000,bootstrap_type='Bernoulli',subsample=.8,random_strength=1.,border_count=32,task_type='CPU',thread_count=16,allow_writing_files=False,use_best_model=True,od_type='Iter',od_wait=150,verbose=False,random_seed=seed)
    et=dict(n_estimators=1000,criterion='squared_error',max_depth=None,min_samples_split=2,min_weight_fraction_leaf=0.,max_leaf_nodes=None,min_impurity_decrease=0.,bootstrap=False,oob_score=False,warm_start=False,ccp_alpha=0.,n_jobs=16,random_state=seed)
    def fit(family,params):
        random.seed(seed); np.random.seed(seed)
        if family=='catboost':
            m=CatBoostRegressor(**params); m.fit(X,y,eval_set=(V,vy),callbacks=[Progress(reporter)])
        else:
            with threadpool_limits(limits=1): m=ExtraTreesRegressor(**params).fit(XF,y)
        return m
    def predict(m,f):
        with threadpool_limits(limits=1): return m.predict(V if f=='catboost' else VF)
    def metrics(p):
        return dict(R2=float(r2_score(vy,p)),Pearson_r=float(np.corrcoef(vy,p)[0,1]),RMSE=float(np.sqrt(mean_squared_error(vy,p))),MAE=float(np.mean(abs(vy-p))),bias=float(np.mean(p-vy)),prediction_range=[float(p.min()),float(p.max())],prediction_std=float(p.std()))
    best={}
    for f,grid in [('catboost',[dict(cb,depth=d,l2_leaf_reg=l) for d,l in [(3,10),(3,30),(4,10),(4,30)]]),('extratrees',[dict(et,min_samples_leaf=l,max_features=q) for l,q in [(2,.5),(2,1.),(5,.5),(5,1.),(10,.5),(10,1.)]])]:
        if f=='extratrees': reporter.report_unavailable('ExtraTrees exposes no epoch/batch callbacks; each fixed 1000-tree forest is fitted completely.')
        for params in grid:
            start=time.monotonic()
            try:
                m=fit(f,params); p=predict(m,f); met=metrics(p)
                trees=m.tree_count_ if f=='catboost' else len(m.estimators_)
                row=dict(family=f,parameters=params,seconds=time.monotonic()-start,train_rows=len(y),trees=trees,metrics=met,validation_predictions=p.tolist())
                log['candidates'].append(row); save()
                key=(met['R2'],-trees,-params['depth'],params['l2_leaf_reg']) if f=='catboost' else (met['R2'],params['min_samples_leaf'],-params['max_features'])
                if f not in best or key>best[f]['key']: best[f]=dict(key=key,params=params,p=p.copy())
                reporter.report_epoch(epoch=trees,val_metric_name='R2',val_metric=met['R2'])
                del m; gc.collect()
            except Exception as e:
                log['failure']=repr(e); save(); raise
    preds={f:b['p'] for f,b in best.items()}; preds['equal_average']=.5*preds['catboost']+.5*preds['extratrees']
    alternatives={f:metrics(p) for f,p in preds.items()}
    mode=max(['catboost','extratrees','equal_average'],key=lambda f:alternatives[f]['R2'])
    components=['catboost','extratrees'] if mode=='equal_average' else [mode]
    log.update(alternatives=alternatives,mode=mode,selection_reason='Maximum global validation R2 among independently selected families and predeclared equal average; exact ties prefer single model then CatBoost.',residual_correlation=float(np.corrcoef(vy-preds['catboost'],vy-preds['extratrees'])[0,1]),refits=[])
    deployed={}
    for f,b in best.items():
        start=time.monotonic(); m=fit(f,b['params']); p=predict(m,f)
        np.testing.assert_allclose(p,b['p'],atol=1e-12,rtol=1e-12)
        log['refits'].append(dict(family=f,seconds=time.monotonic()-start,max_difference=float(np.max(abs(p-b['p'])))))
        if f in components:
            deployed[f]=m
        del m; gc.collect()
    config=dict(mode=mode,components=components,constituent_count=len(components),weights=[.5,.5] if len(components)==2 else [1.],seed=seed,parameters={f:best[f]['params'] for f in components},dependencies=log['dependencies'])
    joblib.dump(dict(config=config,models=deployed),a.artifacts_dir/'model.joblib',compress=3)
    (a.artifacts_dir/'config.json').write_text(json.dumps(config,indent=2))
    save(); reporter.report_epoch(epoch=1,val_metric_name='R2',val_metric=alternatives[mode]['R2'])
    print(json.dumps(dict(mode=mode,alternatives=alternatives)))
if __name__=='__main__': main()
