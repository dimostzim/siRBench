"""Training-reference-only polynomial kernel prediction (float64 CPU)."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='16'
import numpy as np
from normalization import transform

def polynomial(q, beta, eta):
    """Elementwise polynomial of signed query-to-training dot products."""
    if beta < 0 or eta < 0:
        raise ValueError('Kernel weights must be nonnegative')
    q = np.asarray(q, dtype=np.float64)
    return q + beta*q*q + eta*q*q*q

def predict(x,state,batch_size=128):
    if batch_size < 1: raise ValueError('Invalid batch size')
    out=[]
    for start in range(0,len(x),batch_size):
        z=transform(x[start:start+batch_size],state['mu'],float(state['s2']))
        q=z@state['references'].T
        k=polynomial(q, float(state['beta']), float(state['eta']))
        k-=k.mean(axis=1,keepdims=True)
        k-=state['column_means']; k+=state['grand_mean']
        out.append(float(state['label_mean'])+k@state['dual'])
    p=np.concatenate(out) if out else np.empty(0,dtype=np.float64)
    if not np.isfinite(p).all(): raise ValueError('Nonfinite predictions')
    return p
