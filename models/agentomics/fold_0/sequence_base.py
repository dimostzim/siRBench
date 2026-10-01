"""Stateless siRNA representation; no fitted preprocessing or label access."""
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

GUIDE = 'ACGU'
CONTEXT = 'ACGTX'
PAIRS = [''.join(x) for x in product(GUIDE, repeat=2)]
TRIPLES = [''.join(x) for x in product(GUIDE, repeat=3)]
FLANK_POSITIONS = list(range(19)) + list(range(38, 57))
FOURS = [''.join(x) for x in product(GUIDE, repeat=4)]
WIDTH = 3338
NONADJACENT_POSITIONS = [(i,j) for i in range(19) for j in range(i+2,19)]

def feature_schema(w_pair=0):
    if w_pair not in (0, 0.25, 1, 4):
        raise ValueError('Pair weight outside prescribed grid')
    w_local, w_global, w_four = 0.25, 4, 1
    return dict(version='sirna_engineered_v3_nonadjacent', dtype='float64', width=WIDTH,
                w_pair=float(w_pair),
                w_local=float(w_local), w_global=float(w_global), w_four=float(w_four),
                blocks=[
                    dict(name='guide_position', start=0, stop=76, positions=list(range(19)), channels=list(GUIDE), scale=1.0),
                    dict(name='flank_position', start=76, stop=266, positions=FLANK_POSITIONS, channels=list(CONTEXT), scale=0.5),
                    dict(name='guide_adjacent_pair_position', start=266, stop=554, positions=list(range(18)), channels=PAIRS, scale=float(np.sqrt(w_local))),
                    dict(name='guide_overlapping_kmer_counts', start=554, stop=634, channels=PAIRS+TRIPLES, pair_divisor=float(np.sqrt(18)), triple_divisor=float(np.sqrt(17)), scale=float(np.sqrt(w_global))),
                    dict(name='guide_overlapping_fourmer_counts', start=634, stop=890, channels=FOURS, divisor=4.0, scale=float(np.sqrt(w_four))),
                    dict(name='guide_nonadjacent_pair_position', start=890, stop=3338, positions=NONADJACENT_POSITIONS, channels=PAIRS, scale=float(np.sqrt(w_pair*18/153)), scale_formula='sqrt(w_pair * 18 / 153)')],
                ordering='position-major within positional blocks; channels in specified order',
                labels='Untransformed original continuous efficacy',
                preprocessing='Stateless; no centering, standardization, feature selection or imputation')

def validate_input(frame):
    for col in ('id', 'siRNA', 'extended_mRNA'):
        if col not in frame:
            raise ValueError('Missing column: '+col)
    ids = frame['id']
    if not ids.map(lambda x: isinstance(x, str) and len(x)>0).all() or ids.duplicated().any():
        raise ValueError('IDs must be unique nonempty strings')
    for g, c in zip(frame.siRNA, frame.extended_mRNA):
        if not isinstance(g,str) or len(g)!=19 or not set(g)<=set(GUIDE):
            raise ValueError('Invalid guide')
        if not isinstance(c,str) or len(c)!=57 or not set(c)<=set(CONTEXT):
            raise ValueError('Invalid context')
        if c[19:38] != g.translate(str.maketrans('ACGU','TGCA'))[::-1]:
            raise ValueError('Context center is not reverse complement of guide')

def load_input(input_dir):
    frame = pd.read_csv(Path(input_dir)/'data.csv', dtype=str, keep_default_na=False)
    validate_input(frame)
    return frame

def load_split(split_dir):
    frame = load_input(Path(split_dir)/'input')
    labels = pd.read_csv(Path(split_dir)/'labels.csv', dtype={'id':str}, keep_default_na=False)
    if 'id' not in labels or labels.id.duplicated().any() or (labels.id=='').any():
        raise ValueError('Invalid label IDs')
    if set(labels.id) != set(frame.id):
        raise ValueError('Input/label ID sets differ')
    col = 'numeric_label' if 'numeric_label' in labels else 'label'
    y = pd.to_numeric(labels.set_index('id').loc[frame.id,col], errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(y).all() or ((y<0)|(y>1)).any():
        raise ValueError('Invalid efficacy labels')
    return frame, y

def encode(frame, w_pair=0):
    feature_schema(w_pair)
    w_local, w_global, w_four = 0.25, 4, 1
    validate_input(frame)
    out = np.zeros((len(frame), WIDTH), dtype=np.float64)
    gp = {x:i for i,x in enumerate(GUIDE)}
    cp = {x:i for i,x in enumerate(CONTEXT)}
    dp = {x:i for i,x in enumerate(PAIRS)}
    tp = {x:i for i,x in enumerate(TRIPLES)}
    fp = {x:i for i,x in enumerate(FOURS)}
    for row,(g,c) in enumerate(zip(frame.siRNA,frame.extended_mRNA)):
        for pos,base in enumerate(g):
            out[row,4*pos+gp[base]]=1
        for j,pos in enumerate(FLANK_POSITIONS):
            out[row,76+5*j+cp[c[pos]]]=0.5
        for pos in range(18):
            pair=dp[g[pos:pos+2]]
            out[row,266+16*pos+pair]=np.sqrt(w_local)
            out[row,554+pair]+=1
        for pos in range(17):
            out[row,570+tp[g[pos:pos+3]]]+=1
        for pos in range(16):
            out[row,634+fp[g[pos:pos+4]]]+=1
        for p,(i,j) in enumerate(NONADJACENT_POSITIONS):
            out[row,890+16*p+dp[g[i]+g[j]]]=np.sqrt(w_pair*18/153)
    out[:,634:890]*=np.sqrt(w_four)/4.0
    out[:,554:570]*=np.sqrt(w_global)/np.sqrt(18)
    out[:,570:634]*=np.sqrt(w_global)/np.sqrt(17)
    return out
