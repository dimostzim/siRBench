"""Stateless physical folding descriptors; no experimental efficacy training."""
import re
import numpy as np
import RNA

SETTINGS = dict(package='ViennaRNA', version='2.7.2', parameters='RNA Turner 2004', temperature=37.0, dangles=2, noLP=False, probability_tolerance=1e-7)
if RNA.__version__ != SETTINGS['version']:
    raise RuntimeError('Expected ViennaRNA '+SETTINGS['version'])
RNA.params_load_RNA_Turner2004()
NAMES = ([f'guide_unpaired_{i:02d}' for i in range(19)]
         + [f'central_context_unpaired_{i:02d}' for i in range(19)]
         + ['left_observed_unpaired_mean','right_observed_unpaired_mean']
         + ['guide_first5_unpaired_mean','guide_last5_unpaired_mean','central_first5_unpaired_mean','central_last5_unpaired_mean']
         + ['guide_mfe','guide_ensemble_energy','context_fragment_mfe_sum','context_fragment_ensemble_energy_sum'])

def fold(sequence):
    if set(sequence)-set('ACGU'): raise ValueError('Invalid RNA fragment')
    if len(sequence)<=3: return np.ones(len(sequence)),0.0,0.0
    md=RNA.md(); md.temperature=37.; md.dangles=2; md.noLP=False
    fc=RNA.fold_compound(sequence,md)
    _,mfe=fc.mfe()
    fc.exp_params_rescale(mfe)
    _,ensemble=fc.pf()
    bpp=fc.bpp()
    paired=np.zeros(len(sequence))
    for i in range(1,len(sequence)+1):
        for j in range(i+1,len(sequence)+1):
            p=bpp[i][j]
            paired[i-1]+=p; paired[j-1]+=p
    unpaired=1-paired
    if not np.isfinite(unpaired).all() or not np.isfinite([mfe,ensemble]).all(): raise ValueError('Nonfinite folding output')
    if unpaired.min() < -1e-7 or unpaired.max()>1+1e-7: raise ValueError('Material probability error')
    return np.clip(unpaired,0,1),float(mfe),float(ensemble)

def context_fold(context):
    if len(context)!=57 or set(context)-set('ACGTX'): raise ValueError('Invalid context')
    # X coordinates remain masked, never spliced out or treated as bases.
    probabilities=np.full(57,np.nan); mfe=ensemble=0.
    for match in re.finditer('[ACGT]+',context):
        p,m,e=fold(match.group().replace('T','U'))
        probabilities[match.start():match.end()]=p
        mfe+=m; ensemble+=e
    return probabilities,mfe,ensemble

def observed_mean(p):
    valid=p[np.isfinite(p)]
    return float(valid.mean()) if len(valid) else 0.

def transform(frame):
    output=np.empty((len(frame),48),dtype=np.float64)
    for i,(guide,context) in enumerate(zip(frame.siRNA,frame.extended_mRNA)):
        if len(guide)!=19 or set(guide)-set('ACGU'): raise ValueError('Invalid guide')
        if context[19:38]!=guide.replace('U','T').translate(str.maketrans('ACGT','TGCA'))[::-1]: raise ValueError('Central mismatch')
        g,gm,ge=fold(guide); c,cm,ce=context_fold(context); central=c[19:38]
        output[i]=np.r_[g,central,observed_mean(c[:19]),observed_mean(c[38:]),g[:5].mean(),g[-5:].mean(),central[:5].mean(),central[-5:].mean(),gm,ge,cm,ce]
    if not np.isfinite(output).all(): raise ValueError('Nonfinite structural features')
    return output
