import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='16'
from sequence_model import device_setup,predict
from architecture import *
from features import read_inputs,read_labels,raw_transform,ThermoScaler
from helpers.training_reporter import TrainingReporter
import argparse,json,random,time,gc,importlib.metadata,tempfile
from sklearn.model_selection import KFold,train_test_split
from calibration import fit_oof_affine,apply_affine
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

def metrics(y,p):
    return dict(R2=float(r2_score(y,p)),Pearson=float(np.corrcoef(y,p)[0,1]),RMSE=float(mean_squared_error(y,p)**.5),MAE=float(mean_absolute_error(y,p)),bias=float(np.mean(p-y)))
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def fit_cnn(factory,tx,vx,ty,vy,seed,device,out,reporter):
    seed_all(seed); start=time.time()
    tx=[torch.from_numpy(x) for x in tx.values()]; vx=[torch.from_numpy(x) for x in vx.values()]
    model=factory(ty.mean()).to(device)
    optimizer=torch.optim.AdamW([{'params':[v for v in model.parameters() if v.ndim>1],'weight_decay':.01},{'params':[v for v in model.parameters() if v.ndim<=1],'weight_decay':0}],lr=.001,betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=1e-5)
    generator=torch.Generator().manual_seed(seed)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tx,torch.tensor(ty,dtype=torch.float32)),batch_size=64,shuffle=True,generator=generator)
    best=-float('inf'); stale=0; history=[]
    for epoch in range(1,101):
        model.train(); total=0
        for batch in loader:
            batch=[b.to(device) for b in batch]; optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(model(*batch[:-1]),batch[-1]); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); total+=loss.item()*len(batch[-1])
        scheduler.step(); p=predict(model,vx,device); score=float(r2_score(vy,p))
        if score>best:
            best=score; best_epoch=epoch; stale=0; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        history.append(dict(epoch=epoch,train_loss=total/len(ty),val_R2=score))
        reporter.report_epoch(epoch,train_loss=total/len(ty),val_loss=float(mean_squared_error(vy,p)),val_metric_name='R2',val_metric=score,early_stopping_patience_remaining=20-stale)
        if stale>=20: break
    model.load_state_dict(state); p=predict(model,vx,device); torch.save(state,out)
    result=dict(best_epoch=best_epoch,stopping_epoch=epoch,seconds=time.time()-start,history=history,validation=metrics(vy,p))
    del model,optimizer,scheduler,loader,batch,loss; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return p,result

class TreeReporter:
    def __init__(self,reporter): self.reporter=reporter
    def after_iteration(self,info):
        if info.iteration == 1 or info.iteration % 100 == 0:
            self.reporter.report_epoch(info.iteration,train_loss=float(info.metrics['learn']['RMSE'][-1])**2,val_metric_name='R2',val_metric=float(info.metrics['validation']['R2'][-1]))
        return True

def subset(raw, indices):
    return {'sequence':{k:v[indices] for k,v in raw['sequence'].items()},'thermo_raw':raw['thermo_raw'][indices],'tree':raw['tree'][indices]}

def fit_calibration(tr,ty,ids,seed,device,reporter,audit):
    oof=np.full(len(ty),np.nan); coverage=np.zeros(len(ty),dtype=int); logs=[]
    with tempfile.TemporaryDirectory(prefix='temporary_cv_',dir=audit.parent) as temporary:
        checkpoint=Path(temporary)/'fold.pt'
        for j,(fitting,heldout) in enumerate(KFold(n_splits=5,shuffle=True,random_state=seed).split(ty)):
            optimization,stopping=train_test_split(fitting,test_size=.15,shuffle=True,random_state=seed+100+j)
            txraw,sxraw,qxraw=[subset(tr,i) for i in (optimization,stopping,heldout)]
            scaler=ThermoScaler().fit(txraw['thermo_raw']); ps=[]; log={'fold':j,'stop_seed':seed+100+j,'optimization_count':len(optimization),'stopping_count':len(stopping),'heldout_count':len(heldout)}
            for name,factory in [('sequence',make_sequence_model),('thermo',make_thermodynamic_model)]:
                tx,sx,qx=[dict(r['sequence']) for r in (txraw,sxraw,qxraw)]
                if name=='thermo':
                    for x,r in zip((tx,sx,qx),(txraw,sxraw,qxraw)): x['thermo']=scaler.transform(r['thermo_raw'])
                _,log[name]=fit_cnn(factory,tx,sx,ty[optimization],ty[stopping],seed,device,checkpoint,reporter)
                model=factory(0).to(device); model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
                ps.append(predict(model,[torch.from_numpy(x) for x in qx.values()],device)); del model
            seed_all(seed); tree=make_tree_model(seed); start=time.time()
            tree.fit(txraw['tree'],ty[optimization],eval_set=(sxraw['tree'],ty[stopping]),callbacks=[TreeReporter(reporter)])
            ps.append(tree.predict(qxraw['tree'])); log['tree']={'tree_count':tree.tree_count_,'seconds':time.time()-start}
            oof[heldout]=combine_predictions(*ps); coverage[heldout]+=1; logs.append(log)
    assert np.all(coverage==1) and np.isfinite(oof).all()
    calibration=fit_oof_affine(oof,ty)
    (audit/'calibration_fit.json').write_text(json.dumps({'folds':logs,'calibration':calibration,'raw_oof':metrics(ty,oof),'calibrated_oof_in_sample':metrics(ty,apply_affine(oof,calibration))},indent=2))
    pd.DataFrame({'id':ids,'oof_mean':oof}).to_csv(audit/'oof_predictions.csv',index=False)
    return calibration

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-data',required=True); ap.add_argument('--validation-data',required=True); ap.add_argument('--artifacts-dir',required=True); a=ap.parse_args()
    out=Path(a.artifacts_dir); out.mkdir(parents=True,exist_ok=True)
    audit=out.parent/('audit_'+out.name); audit.mkdir(exist_ok=True)
    seed=int(os.environ.get('AGENTOMICS_TRAIN_SEED','0')); device=device_setup(); reporter=TrainingReporter()
    config=dict(seed=seed,predictor_count=3,weights=[1/3]*3,cnn=CNN_TRAINING,tree=CATBOOST_PARAMETERS,sequence_dropout=.4,thermo_hidden=16,thermo_dropout=.5,policy='iteration 17 of 20 (first 3 exploration); five-fold training-only affine calibration then three final fits; no extra candidates; original search budget 20 iterations',dependencies={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scikit-learn','catboost','ViennaRNA']})
    (audit/'policy.json').write_text(json.dumps(config,indent=2))
    tf=read_inputs(Path(a.train_data)/'input'); vf=read_inputs(Path(a.validation_data)/'input')
    ty=read_labels(Path(a.train_data)/'labels.csv',tf.id)
    tr=raw_transform(tf)
    calibration=fit_calibration(tr,ty,tf.id,seed,device,reporter,audit)
    (out/'calibration.json').write_text(json.dumps(calibration,indent=2))
    vy=read_labels(Path(a.validation_data)/'labels.csv',vf.id)
    va=raw_transform(vf); scaler=ThermoScaler().fit(tr['thermo_raw']); scaler.save(out/'scaler.npz')
    ps=[]; results={}
    for name,factory in [('sequence',make_sequence_model),('thermo',make_thermodynamic_model)]:
        tx=dict(tr['sequence']); vx=dict(va['sequence'])
        if name=='thermo': tx['thermo']=scaler.transform(tr['thermo_raw']); vx['thermo']=scaler.transform(va['thermo_raw'])
        p,result=fit_cnn(factory,tx,vx,ty,vy,seed,device,out/(name+'.pt'),reporter); ps.append(p); results[name]=result
    start=time.time(); seed_all(seed); tree=make_tree_model(seed)
    tree.fit(tr['tree'],ty,eval_set=(va['tree'],vy),callbacks=[TreeReporter(reporter)])
    tree.save_model(str(out/'tree.cbm')); ps.append(tree.predict(va['tree']))
    results['tree']=dict(tree_count=tree.tree_count_,best_iteration=tree.best_iteration_,seconds=time.time()-start,validation=metrics(vy,ps[-1]))
    raw=combine_predictions(*ps); results['uncalibrated']=metrics(vy,raw)
    p=apply_affine(raw,calibration); results['ensemble']=metrics(vy,p)
    results['calibration']=calibration
    results['prediction_summary']=dict(mean=float(p.mean()),std=float(p.std()),minimum=float(p.min()),maximum=float(p.max()))
    results['residual_correlations']=np.corrcoef(np.array(ps)-vy).tolist()
    results['failures']=[]; results['selection_reason']='Prespecified fixed three-view mean with training-only OOF affine calibration frozen before validation evaluation; constituent best global validation R2 restored.'
    config['label_mean']=float(ty.mean()); (out/'config.json').write_text(json.dumps(config,indent=2))
    (audit/'training.json').write_text(json.dumps(results,indent=2))
    pd.DataFrame(dict(id=vf.id,sequence=ps[0],thermo=ps[1],tree=ps[2],prediction=p)).to_csv(audit/'validation_predictions.csv',index=False)
    print('FINAL',results['ensemble'])
if __name__=='__main__': main()
