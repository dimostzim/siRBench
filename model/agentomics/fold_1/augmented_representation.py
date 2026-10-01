"""1012-column encoder and train-only normalization. Zero-weight inference can prune folding."""
import json
from pathlib import Path
import numpy as np
from representation import BlockNormalizer, encode as base_encode

STRUCTURE_WEIGHTS=(0.,0.05,0.2,0.8)

def encode(frame):
    import folding_features
    return np.concatenate([base_encode(frame),folding_features.transform(frame)],axis=1)

class AugmentedNormalizer:
    def __init__(self,structure_weight):
        if structure_weight not in STRUCTURE_WEIGHTS: raise ValueError('Undeclared weight')
        self.structure_weight=float(structure_weight)
        self.base=BlockNormalizer(context_multiplier=1.,motif_multiplier=0.5)

    def fit(self,x):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=1012 or not len(x) or not np.isfinite(x).all(): raise ValueError('Expected nonempty finite (n,1012)')
        self.base.fit(x[:,:964])
        self.structure=None
        if self.structure_weight:
            s=x[:,964:]; mu=s.mean(0); sd=s.std(0)
            active=np.ptp(s,axis=0)>0
            sd[~active]=1.
            z=(s-mu)/sd; z[:,~active]=0.
            v=float(np.mean(np.sum(z*z,axis=1)))
            self.structure=dict(mean=mu.tolist(),std=sd.tolist(),active=active.tolist(),variance=v)
        return self

    def transform(self,x):
        x=np.asarray(x,dtype=np.float64)
        allowed=(964,1012) if not self.structure_weight else (1012,)
        if x.ndim!=2 or x.shape[1] not in allowed or not np.isfinite(x).all(): raise ValueError('Invalid input features')
        b=self.base.transform(x[:,:964])
        if not self.structure_weight: return b
        s=self.structure
        z=(x[:,964:]-np.array(s['mean']))/np.array(s['std']); z[:,~np.array(s['active'])]=0.
        if s['variance']==0: z[:]=0.
        else: z*=np.sqrt(self.structure_weight/max(s['variance'],1e-12))
        if not np.isfinite(z).all(): raise ValueError('Invalid normalized features')
        return np.concatenate([b,z],axis=1)

    def encode_transform(self,frame):
        return self.transform(encode(frame) if self.structure_weight else base_encode(frame))

    def save(self,directory):
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
        self.base.save(directory/'base_normalizer.json')
        (directory/'structure_normalizer.json').write_text(json.dumps(dict(version=1,structure_weight=self.structure_weight,structure=self.structure),indent=2)+'\n')

    @classmethod
    def load(cls,directory):
        directory=Path(directory); state=json.loads((directory/'structure_normalizer.json').read_text())
        if state['version']!=1: raise ValueError('Unsupported schema')
        obj=cls(state['structure_weight']); obj.structure=state['structure']; obj.base=BlockNormalizer.load(directory/'base_normalizer.json')
        return obj
