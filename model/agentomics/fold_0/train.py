import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,json,random,time,platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch,scipy,sklearn,catboost
import tree_features as tf
from scipy.linalg import cho_factor,cho_solve
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from helpers.training_reporter import TrainingReporter
import cnn_features as cf
import sequence_features as kf
from normalization import fit_normalization,transform,representation_schema
from kernel_model import polynomial,predict as kernel_predict
from network import SequenceRegressor

def metrics(y,p):
    return dict(R2=float(r2_score(y,p)),Pearson_r=float(np.corrcoef(y,p)[0,1]),RMSE=float(mean_squared_error(y,p)**.5),MAE=float(mean_absolute_error(y,p)))
def predict(model,x,stats):
    model.eval()
    with torch.inference_mode():
        z=torch.cat([model(*(a[i:i+256] for a in x)) for i in range(0,len(x[0]),256)]).cpu().numpy()
    return cf.inverse_targets(z,stats)
def main():
    parser=argparse.ArgumentParser()
    for k in ['train-data','validation-data','artifacts-dir']: parser.add_argument('--'+k,required=True)
    args=parser.parse_args(); out=Path(args.artifacts_dir); out.mkdir(parents=True,exist_ok=True); root=out.parent
    policy=json.loads((Path(__file__).parent/'search_policy.json').read_text()); seed=int(os.environ.get('AGENTOMICS_TRAIN_SEED','0'))
    policy['actual_seed']=seed
    (root/'search_policy.json').write_text(json.dumps(policy,indent=2))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(min(1.,7*1024**3/torch.cuda.get_device_properties(0).total_memory),0)
    train=cf.read_inputs(Path(args.train_data)/'input'); val=cf.read_inputs(Path(args.validation_data)/'input')
    y=cf.read_labels(args.train_data,train); vy=cf.read_labels(args.validation_data,val)
    reporter=TrainingReporter(); start=time.monotonic()
    reporter.report_unavailable('Component A uses one float64 Cholesky direct solve, without epochs or batch callbacks.')
    x=kf.encode(train); xv=kf.encode(val); mu,s2=fit_normalization(x); z=transform(x,mu,s2)
    k=polynomial(z@z.T,1.,.3); col=k.mean(0); grand=k.mean(); kc=k-col[None,:]-k.mean(1)[:,None]+grand
    system=kc+3*np.eye(len(y)); yc=y-y.mean(); dual=cho_solve(cho_factor(system,lower=True),yc)
    residual=float(np.linalg.norm(system@dual-yc)/max(np.linalg.norm(yc),1e-15))
    if residual>1e-8: raise ValueError('Excessive solve residual')
    kernel=dict(references=z,mu=mu,s2=s2,dual=dual,label_mean=y.mean(),column_means=col,grand_mean=grand,beta=1.,eta=.3,alpha=3.)
    pa=kernel_predict(xv,kernel); ma=metrics(vy,pa)
    reporter.report_epoch(epoch=1,val_metric_name='R2',val_metric=ma['R2'])
    kernel_seconds=time.monotonic()-start
    stats=cf.fit_label_stats(y); device=torch.device('cuda:0')
    x=tuple(torch.from_numpy(a).to(device) for a in cf.encode(train)[:3]); vx=tuple(torch.from_numpy(a).to(device) for a in cf.encode(val)[:3]); target=torch.from_numpy(cf.standardize(y,stats)).to(device)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    generator=torch.Generator().manual_seed(seed); cfg=dict(channels=32,dropout=.1); model=SequenceRegressor(**cfg).to(device)
    opt=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if not n.endswith('bias')],'weight_decay':.01},{'params':[p for n,p in model.named_parameters() if n.endswith('bias')],'weight_decay':0}],lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=.00001)
    best=-float('inf'); stale=0; epochs=[]; start=time.monotonic()
    for epoch in range(1,101):
        model.train(); total=0.
        for idx in torch.randperm(len(y),generator=generator).split(64):
            idx=idx.to(device); opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(model(*(a[idx] for a in x)),target[idx]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); total+=loss.item()*len(idx)
        scheduler.step(); p=predict(model,vx,stats); score=float(r2_score(vy,p))
        if score>best:
            best=score; stale=0; best_epoch=epoch; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        reporter.report_epoch(epoch,train_loss=total/len(y),val_loss=float(np.mean((p-vy)**2)),val_metric_name='R2',val_metric=score,early_stopping_patience_remaining=20-stale)
        epochs.append(dict(epoch=epoch,train_loss=total/len(y),validation_R2=score))
        if stale>=20: break
    model.load_state_dict(state); pb=predict(model,vx,stats); pm=(pa+pb)*.5
    cnn_seconds=time.monotonic()-start
    predictions={'A':pa,'B':pb,'mean':pm}
    components={'A':['A'],'B':['B'],'mean':['A','B']}
    scores={n:metrics(vy,p) for n,p in predictions.items()}
    tree_reports={}; trees={}
    tx=tf.encode(train); tv=tf.encode(val)
    for configuration in policy['C']['configurations']:
        name=configuration['name']; params={k:v for k,v in configuration.items() if k!='name'}
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        start=time.monotonic()
        class Progress:
            def after_iteration(self,info):
                if info.iteration % 100 == 0:
                    rmse=info.metrics['validation']['RMSE'][-1]
                    reporter.report_epoch(info.iteration,train_loss=info.metrics['learn']['RMSE'][-1]**2,val_loss=rmse**2,val_metric_name='R2',val_metric=1-rmse**2/np.var(vy))
                return True
        tree=catboost.CatBoostRegressor(**policy['C']['fixed_parameters'],**params,random_seed=seed,verbose=False)
        tree.fit(tx,y,eval_set=(tv,vy),**policy['C']['fit_parameters'],callbacks=[Progress()])
        pc=tree.predict(tv); trees[name]=tree
        predictions[name]=pc; components[name]=[name]
        combo='mean_'+name; predictions[combo]=(pa+pb+pc)/3; components[combo]=['A','B',name]
        scores[name]=metrics(vy,pc); scores[combo]=metrics(vy,predictions[combo])
        tree_reports[name]=dict(config=tree.get_params(),tree_count=tree.tree_count_,seconds=time.monotonic()-start,training=metrics(y,tree.predict(tx)),validation=scores[name])
        reporter.report_epoch(tree.tree_count_,val_metric_name='R2',val_metric=scores[name]['R2'])
    def rank(n):
        c=next((q for q in policy['C']['configurations'] if q['name'] in components[n]),None)
        return (scores[n]['R2'],-len(components[n]),n=='A',-c['depth'] if c else 0,c['l2_leaf_reg'] if c else 0)
    selected=max(scores,key=rank); chosen=components[selected]
    torch.save(dict(kernel=kernel if 'A' in chosen else None,cnn=dict(state_dict=state,config=cfg,label_stats=stats,seed=seed) if 'B' in chosen else None),out/'model.pt')
    for n in chosen:
        if n in trees: trees[n].save_model(str(out/'catboost.cbm'))
    meta=dict(selected=selected,components=chosen,constituent_count=len(chosen),seed=seed,weights=[1/len(chosen)]*len(chosen))
    (out/'model.json').write_text(json.dumps(meta,indent=2))
    (out/'schemas.json').write_text(json.dumps(dict(kernel=representation_schema(),cnn=cf.SCHEMA,tree=tf.feature_schema()),indent=2))
    versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,catboost=catboost.__version__)
    report=dict(**meta,versions=versions,validation=scores,trees=tree_reports,train_count=len(y),validation_count=len(vy),kernel_residual=residual,kernel_seconds=kernel_seconds,cnn_seconds=cnn_seconds,best_epoch=best_epoch,epochs=epoch,peak_gpu_bytes=torch.cuda.max_memory_allocated(),selection_reason='Maximum global validation R2 among exactly fifteen predeclared alternatives; fixed equal weights only.',failures=[])
    (root/'training_report.json').write_text(json.dumps(report,indent=2)); pd.DataFrame(epochs).to_csv(root/'epoch_metrics.csv',index=False)
    pd.DataFrame(dict(id=val.id,**predictions,prediction=predictions[selected])).to_csv(root/'validation_predictions.csv',index=False)
    print(json.dumps(dict(selected=selected,validation=scores)))
if __name__=='__main__': main()
