"""Training-only block normalization for the copied stateless encoder."""
import json
import re
from pathlib import Path
import numpy as np
import features

# Match feature names, never infer regional locations from column offsets.
BLOCKS = (
 ('guide_positional', r'guide_\d{2}_[ACGT]',76,1.0),
 ('context_positional',r'context_\d{2}_[ACGTX]',285,0.25),
 ('guide_dimers',r'guide_dimer_\d{2}_[ACGT]{2}',288,0.5),
 ('guide_kmers',r'guide_k[123]_freq_[ACGT]+',84,0.75),
 ('left_kmers',r'left_k[123]_freq_[ACGT]+',84,0.125),
 ('right_kmers',r'right_k[123]_freq_[ACGT]+',84,0.125),
 ('coverage',r'(guide|left|right)_k[123]_coverage',9,0.05),
 ('rolling_gc',r'guide_gc_w[357]_p\d{2}',45,0.5),
 ('gc_summaries',r'guide_(gc|gc_squared|first5_gc|last5_gc|first5_minus_last5_gc)',5,0.25),
 ('homopolymers',r'guide_max_run_[ACGT]',4,0.1),
)

def block_spec(context_multiplier=1.0, motif_multiplier=1.0):
    if not all(np.isfinite(m) and m >= 0 for m in (context_multiplier, motif_multiplier)):
        raise ValueError('Multipliers must be finite and nonnegative')
    names = list(features.SCHEMA['feature_names'])
    blocks=[]
    for name,pattern,count,weight in BLOCKS:
        indices=[i for i,n in enumerate(names) if re.fullmatch(pattern,n)]
        if len(indices)!=count: raise ValueError('Invalid block '+name)
        if name in ('context_positional','left_kmers','right_kmers','coverage'):
            weight *= context_multiplier
        elif name in ('guide_dimers','guide_kmers'):
            weight *= motif_multiplier
        blocks.append(dict(name=name,indices=indices,weight=weight))
    if sorted(i for b in blocks for i in b['indices'])!=list(range(964)):
        raise ValueError('Blocks must partition the schema')
    return blocks

def encode(frame):
    # Preserve iteration-8 feature values exactly, then promote for SVR arithmetic.
    return features.transform(frame).astype(np.float64)

class BlockNormalizer:
    def __init__(self, context_multiplier=1.0, motif_multiplier=1.0):
        block_spec(context_multiplier, motif_multiplier)
        self.context_multiplier = float(context_multiplier)
        self.motif_multiplier = float(motif_multiplier)

    def _check(self,x):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=964 or not np.isfinite(x).all():
            raise ValueError('Expected finite (n,964) matrix')
        return x

    def fit(self,x):
        x=self._check(x)
        if len(x)==0: raise ValueError('Cannot fit empty training set')
        self.state={'version':2,'feature_names':list(features.FEATURE_NAMES),
                    'context_multiplier':self.context_multiplier,
                    'motif_multiplier':self.motif_multiplier,
                    'training_rows':len(x),'blocks':block_spec(self.context_multiplier,self.motif_multiplier)}
        for b in self.state['blocks']:
            xb=x[:,b['indices']]
            mu=xb.mean(axis=0)
            v=float(np.mean(np.sum((xb-mu)**2,axis=1)))
            b.update(mean=mu.tolist(),variance=v,zero_block=(v==0.0 or b['weight']==0.0))
        return self

    def transform(self,x):
        x=self._check(x)
        if not hasattr(self,'state'): raise ValueError('Normalizer not fitted')
        z=np.zeros_like(x)
        for b in self.state['blocks']:
            ix=b['indices']
            if not b['zero_block']:
                z[:,ix]=(x[:,ix]-np.asarray(b['mean']))*np.sqrt(b['weight']/max(b['variance'],1e-12))
        if not np.isfinite(z).all(): raise ValueError('Nonfinite normalized features')
        return z

    def save(self,path):
        Path(path).write_text(json.dumps(self.state,indent=2)+'\n')

    @classmethod
    def load(cls,path):
        state=json.loads(Path(path).read_text())
        if state.get('version') != 2: raise ValueError('Unsupported normalizer version')
        obj=cls(state['context_multiplier'],state['motif_multiplier']); obj.state=state
        if obj.state['feature_names']!=list(features.FEATURE_NAMES):
            raise ValueError('Saved schema mismatch')
        for actual,expected in zip(obj.state['blocks'],block_spec(obj.context_multiplier,obj.motif_multiplier),strict=True):
            if any(actual[k]!=expected[k] for k in expected):
                raise ValueError('Saved block definition mismatch')
        return obj
