import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cnn_features as cf
import sequence_features as kf
from kernel_model import predict as kp
from network import SequenceRegressor

def predict_frame(frame,artifacts,batch_size=128):
    artifacts=Path(artifacts); meta=json.loads((artifacts/'model.json').read_text()); outputs=[]
    bundle=torch.load(artifacts/'model.pt',map_location='cpu',weights_only=False)
    if 'A' in meta['components']:
        outputs.append(np.concatenate([kp(kf.encode(frame.iloc[i:i+batch_size]),bundle['kernel'],batch_size) for i in range(0,len(frame),batch_size)]) if len(frame) else np.empty(0))
    if 'B' in meta['components']:
        torch.set_num_threads(4)
        saved=bundle['cnn']
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device.type=='cuda':
            torch.cuda.set_per_process_memory_fraction(min(1.0,(7*1024**3)/torch.cuda.get_device_properties(0).total_memory),device=0)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
        model=SequenceRegressor(**saved['config']); model.load_state_dict(saved['state_dict']); model.to(device).eval()
        chunks=[]
        with torch.inference_mode():
            for i in range(0,len(frame),batch_size):
                arrays=tuple(torch.from_numpy(a).to(device) for a in cf.encode(frame.iloc[i:i+batch_size])[:3])
                chunks.append(model(*arrays).cpu().numpy())
        outputs.append(cf.inverse_targets(np.concatenate(chunks) if chunks else np.empty(0),saved['label_stats']))
    if any(n.startswith('C_') for n in meta['components']):
        import tree_features as tf
        from catboost import CatBoostRegressor
        tree=CatBoostRegressor(); tree.load_model(str(artifacts/'catboost.cbm'))
        outputs.append(np.concatenate([tree.predict(tf.encode(frame.iloc[i:i+batch_size]),thread_count=16) for i in range(0,len(frame),batch_size)]) if len(frame) else np.empty(0))
    p=sum(outputs)/len(outputs)
    if p.shape!=(len(frame),) or not np.isfinite(p).all(): raise ValueError('Invalid predictions')
    return p

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--input',required=True); parser.add_argument('--output',required=True); parser.add_argument('--artifacts-dir',default=str(Path(__file__).resolve().parent/'training_artifacts')); a=parser.parse_args()
    frame=cf.read_inputs(a.input); p=predict_frame(frame,a.artifacts_dir)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame({'id':frame.id,'prediction':p}).to_csv(a.output,index=False)
if __name__=='__main__': main()
