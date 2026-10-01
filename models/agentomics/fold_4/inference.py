import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='16'
import argparse,json,csv
from pathlib import Path
import numpy as np
import joblib
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits
from features import load_input

def predict(input_dir,artifacts):
    artifacts=Path(artifacts); cfg=json.loads((artifacts/'config.json').read_text()); ids,X=load_input(input_dir)
    if not ids: return ids,np.empty(0)
    bundle=joblib.load(artifacts/'model.joblib')
    p=np.zeros(len(ids))
    for f,w in zip(cfg['components'],cfg['weights']):
        if f=='catboost':
            m=bundle['models'][f]; q=m.predict(X,thread_count=16)
        else:
            m=bundle['models'][f]; m.n_jobs=16
            with threadpool_limits(limits=1): q=m.predict(np.ascontiguousarray(X,dtype=np.float32))
        p+=w*q
    if p.shape != (len(ids),) or not np.isfinite(p).all():
        raise ValueError('Model did not produce one finite prediction per input row')
    return ids,p

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--artifacts-dir',default=str(Path(__file__).resolve().parent/'training_artifacts')); a=ap.parse_args()
    ids,p=predict(a.input,a.artifacts_dir)
    with open(a.output,'w',newline='') as out:
        w=csv.writer(out); w.writerow(['id','prediction']); w.writerows(zip(ids,p))
if __name__=='__main__': main()
