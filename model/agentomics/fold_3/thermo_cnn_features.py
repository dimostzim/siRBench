"""Combined independent CNN tensors and train-scaled physical descriptors."""
from pathlib import Path
import json
import numpy as np
import sequence_features as cnn
import thermo_features as thermo
from sequence_features import read_inputs, read_labels

REPRESENTATION={'version':'cnn_plus_thermodynamics_v1','cnn':cnn.REPRESENTATION,'thermodynamics':{'shape':[140],'feature_names':thermo.STRUCTURE_NAMES,'folding':thermo.FOLDING,'scaling':'Training-only float64 mean and population std; scale=max(std,0.1); output float32'},'label_transform':None,'predictive_information':'siRNA and extended_mRNA only'}

class ThermoScaler:
    def fit(self, raw):
        x=np.asarray(raw,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=140 or not len(x) or not np.isfinite(x).all(): raise ValueError('Invalid scaler training data')
        self.mean=x.mean(axis=0); self.scale=np.maximum(x.std(axis=0,ddof=0),0.1)
        return self
    def transform(self, raw):
        x=np.asarray(raw,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=140: raise ValueError('Invalid shape')
        out=((x-self.mean)/self.scale).astype(np.float32)
        if not np.isfinite(out).all(): raise ValueError('Nonfinite scaled features')
        return out
    def save(self,path):
        np.savez(path,mean=self.mean,scale=self.scale)
    @classmethod
    def load(cls,path):
        obj=cls()
        with np.load(path,allow_pickle=False) as data:
            obj.mean=data['mean']; obj.scale=data['scale']
        if obj.mean.shape!=(140,) or obj.scale.shape!=(140,) or not np.isfinite(obj.mean).all() or not np.isfinite(obj.scale).all() or (obj.scale<0.1).any(): raise ValueError('Invalid scaler artifact')
        return obj

def raw_transform(frame):
    out=cnn.transform(frame)
    out['thermo']=thermo.transform(frame)
    return out

def transform(frame,scaler):
    out=raw_transform(frame)
    out['thermo']=scaler.transform(out['thermo'])
    return out

def save_spec(artifact_dir):
    path=Path(artifact_dir); path.mkdir(parents=True,exist_ok=True)
    (path/'representation.json').write_text(json.dumps(REPRESENTATION,indent=2)+'\n')
