import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='16'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,json,random,time,csv,warnings
from pathlib import Path
import numpy as np
import torch,sklearn
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from helpers.training_reporter import TrainingReporter
import features as F
from model import EfficacyCNN

def predict_cnn(model,arrays,state,device):
    model.eval()
    with torch.inference_mode():
        z=[model(*[torch.from_numpy(a[i:i+128]).to(device) for a in arrays]).cpu().numpy() for i in range(0,len(arrays[0]),128)]
    return F.inverse_targets(np.concatenate(z) if z else np.empty(0),state)

def main():
    p=argparse.ArgumentParser()
    for key in ['train-data','validation-data','artifacts-dir']: p.add_argument('--'+key,required=True)
    args=p.parse_args(); out=Path(args.artifacts_dir); out.mkdir(parents=True,exist_ok=True)
    start=time.time(); seed=int(os.getenv('AGENTOMICS_TRAIN_SEED','0'))
    torch.set_num_threads(16); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for training')
    torch.cuda.set_per_process_memory_fraction(min(1.,7*1024**3/torch.cuda.get_device_properties(0).total_memory),device=0)
    device=torch.device('cuda:0')
    ti,tr=F.read_inputs(Path(args.train_data)/'input'); vi,vr=F.read_inputs(Path(args.validation_data)/'input')
    y=F.load_labels(Path(args.train_data)/'labels.csv',ti); vy=F.load_labels(Path(args.validation_data)/'labels.csv',vi)
    state=F.fit_target_transform(y); ta=list(F.transform_tensors(tr).values()); va=list(F.transform_tensors(vr).values())
    ds=torch.utils.data.TensorDataset(*[torch.from_numpy(x) for x in ta],torch.from_numpy(F.transform_targets(y,state)))
    config=json.loads((Path(__file__).parent/'architecture_config.json').read_text())
    reporter=TrainingReporter(); histories=[]; winner_key=None
    with warnings.catch_warnings(record=True) as caught:
        for candidate in config['candidates']:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            loader=torch.utils.data.DataLoader(ds,batch_size=64,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
            model=EfficacyCNN(candidate['dropout']).to(device)
            opt=torch.optim.AdamW(model.optimizer_groups(candidate['weight_decay']),lr=candidate['learning_rate'],betas=(.9,.999),eps=1e-8)
            sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=1e-5)
            best=-float('inf'); stale=0; history=[]
            for epoch in range(1,101):
                model.train(); total=0.
                for batch in loader:
                    batch=[x.to(device) for x in batch]; opt.zero_grad(set_to_none=True)
                    loss=torch.nn.functional.mse_loss(model(*batch[:-1]),batch[-1]); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); total+=loss.item()*len(batch[-1])
                pred=predict_cnn(model,va,state,device); score=float(r2_score(vy,pred)); sched.step()
                if score>best:
                    best=score; stale=0; best_epoch=epoch
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else: stale+=1
                history.append(dict(epoch=epoch,train_loss=total/len(y),R2=score))
                reporter.report_epoch(epoch,train_loss=total/len(y),val_loss=float(mean_squared_error(vy,pred)),val_metric_name='R2',val_metric=score,early_stopping_patience_remaining=20-stale)
                if stale>=20: break
            model.load_state_dict(best_state); restored=predict_cnn(model,va,state,device)
            result=dict(candidate,best_epoch=best_epoch,stopping_epoch=epoch,R2=float(r2_score(vy,restored)),Pearson=float(np.corrcoef(vy,restored)[0,1]))
            histories.append(dict(result,history=history))
            with (out.parent/f"candidate_p{candidate['dropout']}_lr{candidate['learning_rate']}.csv").open('w') as f:
                w=csv.writer(f); w.writerow(['id','prediction']); w.writerows(zip(vi,restored))
            key=(result['R2'],candidate['dropout'],-candidate['learning_rate'],-best_epoch)
            if winner_key is None or key>winner_key: winner_key=key; winner=result; winner_state=best_state; winning_pred=restored
            del model,opt,sched
    torch.save(winner_state,out/'cnn.pt')
    versions={'numpy':np.__version__,'torch':torch.__version__,'scikit-learn':sklearn.__version__}
    (out/'metadata.json').write_text(json.dumps(dict(target=state,seed=seed,config=config,selected=winner,versions=versions,constituent_count=1),indent=2))
    pred=winning_pred
    metrics=dict(R2=float(r2_score(vy,pred)),Pearson=float(np.corrcoef(vy,pred)[0,1]),RMSE=float(mean_squared_error(vy,pred)**.5),MAE=float(mean_absolute_error(vy,pred)))
    log=dict(seed=seed,n_train=len(y),n_validation=len(vy),selected=winner,candidates=histories,metrics=metrics,seconds=time.time()-start,warnings=[str(w.message) for w in caught],failures=[],selection=config['selection'],versions=versions)
    (out.parent/'training_log.json').write_text(json.dumps(log,indent=2))
    with (out.parent/'validation_predictions.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['id','label','prediction']); w.writerows(zip(vi,vy,pred))
    print(json.dumps(metrics))
if __name__=='__main__': main()
