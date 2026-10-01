import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[k]='16'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
torch.set_num_interop_threads(1)
import features as F
from model import EfficacyCNN

def predict(rows,artifacts,batch_size=128):
    torch.set_num_threads(16)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(min(1.,7*1024**3/torch.cuda.get_device_properties(0).total_memory),device=0)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    artifacts=Path(artifacts); meta=json.loads((artifacts/'metadata.json').read_text())
    m=EfficacyCNN(meta['selected']['dropout']); m.load_state_dict(torch.load(artifacts/'cnn.pt',map_location='cpu',weights_only=True)); m.to(device); m.eval(); result=[]
    size=max(1,min(256,batch_size))
    with torch.inference_mode():
        for i in range(0,len(rows),size):
            arrays=F.transform_tensors(rows[i:i+size])
            result.extend(F.inverse_targets(m(*[torch.from_numpy(x).to(device) for x in arrays.values()]).cpu().numpy(),meta['target']))
    result=np.asarray(result); assert np.isfinite(result).all(); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--artifacts-dir',default=str(Path(__file__).resolve().parent/'training_artifacts')); a=p.parse_args()
    ids,rows=F.read_inputs(a.input); pred=predict(rows,a.artifacts_dir)
    if len(pred) != len(ids):
        raise RuntimeError('Prediction count mismatch')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['id','prediction']); w.writerows(zip(ids,pred))
if __name__=='__main__': main()
