import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='16'
import re
import numpy as np
import RNA
from sequence_features import validate_sequences

def fold_run(sequence):
    n=len(sequence)
    if n <= 4: return np.ones(n), 0.0, 0.0
    md=RNA.md(); md.temperature=37.0; md.dangles=2
    fc=RNA.fold_compound(sequence, md)
    _, mfe=fc.mfe()
    fc.exp_params_rescale(mfe)
    _, ensemble=fc.pf()
    p=np.asarray(fc.bpp(),dtype=float)[1:n+1,1:n+1]
    # ViennaRNA bpp() stores only upper-triangle entries.
    p=np.triu(p,1)
    q=1-p.sum(axis=0)-p.sum(axis=1)
    if q.min() < -1e-6 or q.max() > 1+1e-6: raise ValueError('Invalid unpaired probabilities')
    return np.clip(q,0,1),float(mfe),float(ensemble)

def fold_context(context):
    q=np.full(len(context),-1.0); mfe=ensemble=0.0
    for match in re.finditer('[ACGU]+',context.replace('T','U')):
        qi,mi,ei=fold_run(match.group())
        q[match.start():match.end()]=qi; mfe+=mi; ensemble+=ei
    return q,mfe,ensemble

def known_mean(q):
    known=q[q>=0]
    return float(known.mean()) if len(known) else -1.0

def structure_features(guide,context):
    g,gm,ge=fold_run(guide)
    c,cm,ce=fold_context(context)
    n=int((c>=0).sum())
    values=list(g)+list(c)+[gm,ge,gm/19,ge/19,cm,ce,cm/n if n else 0,ce/n if n else 0]
    for k in (2,3,4,5): values.extend([float(g[:k].mean()),float(g[-k:].mean())])
    values.extend(known_mean(c[i:i+19]) for i in (0,19,38))
    for k in (3,5,7):
        values.extend(known_mean(c[19+i:19+i+k]) for i in range(20-k))
    return values

STRUCTURE_NAMES=([f'guide_unpaired_{i}' for i in range(19)]+[f'context_unpaired_{i}' for i in range(57)]+
 ['guide_MFE','guide_ensemble_energy','guide_MFE_per_base','guide_ensemble_energy_per_base','context_run_MFE_sum','context_run_ensemble_energy_sum','context_MFE_per_known_base','context_ensemble_energy_per_known_base']+
 [f'guide_{end}_{k}_mean_unpaired' for k in (2,3,4,5) for end in ('first','last')]+
 [f'context_{region}_mean_unpaired' for region in ('left','central','right')]+
 [f'central_window_{k}_start_{i}_mean_unpaired' for k in (3,5,7) for i in range(20-k)])

def transform(frame):
    validate_sequences(frame)
    result=np.asarray([structure_features(g,c) for g,c in zip(frame.siRNA,frame.extended_mRNA)],dtype=np.float32).reshape(len(frame),140)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite thermodynamics')
    return result

FOLDING = {'package':'ViennaRNA','version':RNA.__version__,'provenance':'Official ViennaRNA general-purpose thermodynamic folding library, PyPI ViennaRNA; no pretrained efficacy model or external efficacy labels.',
 'energy_parameters':'Bundled default RNA Turner 2004 parameters (Mathews et al. 2004; Turner laboratory nearest-neighbor model). No parameter overrides.',
 'temperature_C':37,'dangles':2,'remaining_settings':'Fresh RNA.md() defaults; no global parameter changes permitted.',
 'default_md_snapshot':str(RNA.md()),'procedure':'MFE, exp_params_rescale(MFE), pf(), triangular bpp; q=1-row_sum-column_sum; clip only roundoff within 1e-6.',
 'orientation':'Guide unchanged 5-prime to 3-prime; context T to U.',
 'X_policy':'Fold maximal known runs separately, forbid cross-gap pairing; retain q=-1 at X. Runs length <=4 have q=1 and zero energies.',
 'means':'Known-position marginal q means; all-unknown region/window => -1. Not joint accessibility.',
 'terminal_order':'For k=2,3,4,5: first-k then last-k.',
 'context_energy_normalization':'Sum energies over known runs; divide by total known bases, or zero when none.'}
