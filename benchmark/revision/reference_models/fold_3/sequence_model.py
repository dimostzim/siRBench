import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[name]='16'
import torch
from torch import nn

def device_setup():
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(min(1.,7*1024**3/torch.cuda.get_device_properties(0).total_memory),0)
        return torch.device('cuda:0')
    return torch.device('cpu')

class Model(nn.Module):
    def __init__(self,p,mean=0.):
        super().__init__()
        self.register_buffer('label_mean',torch.tensor(float(mean)))
        self.c3=nn.Conv1d(4,16,3,padding=1); self.c5=nn.Conv1d(4,16,5,padding=2)
        self.guide_head=nn.Sequential(nn.Dropout(p),nn.Linear(608,32),nn.GELU())
        self.context=nn.Conv1d(5,8,5,padding=2)
        self.head=nn.Sequential(nn.Dropout(p),nn.Linear(78,32),nn.GELU(),nn.Dropout(p),nn.Linear(32,1))
        self.direct=nn.Linear(90,1,bias=False)
        nn.init.zeros_(self.direct.weight); nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
    def forward(self,guide,left,right,descriptors):
        g=self.guide_head(torch.cat([torch.nn.functional.gelu(self.c3(guide)),torch.nn.functional.gelu(self.c5(guide))],1).flatten(1))
        contexts=[]
        for f in [left,right]:
            c=torch.nn.functional.gelu(self.context(f)); contexts.extend([c.mean(2),c.amax(2)])
        nonlinear=self.head(torch.cat([g,*contexts,descriptors],1))
        return (self.label_mean+self.direct(torch.cat([guide.flatten(1),descriptors],1))+nonlinear).squeeze(1)

@torch.no_grad()
def predict(model,tensors,device,batch_size=256):
    import numpy as np
    model.eval()
    n=len(tensors[0])
    return np.concatenate([model(*[x[i:i+batch_size].to(device) for x in tensors]).cpu().numpy() for i in range(0,n,batch_size)]) if n else np.empty(0,dtype=np.float32)
