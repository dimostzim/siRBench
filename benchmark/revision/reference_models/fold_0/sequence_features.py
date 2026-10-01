"""Stateless 3,506-column representation. All imports are local or dependencies."""
from itertools import product
import numpy as np
import sequence_base as base
from sequence_base import load_input, load_split, validate_input

WIDTH = 3506
FLANK_WEIGHTS = (0, 0.25, 1, 4, 16)
KMERS = {k: [''.join(t) for t in product('ACGT', repeat=k)] for k in (1,2,3)}

def feature_schema(w_flank=0.25):
    if w_flank not in FLANK_WEIGHTS:
        raise ValueError('Flank weight outside prescribed grid')
    schema = base.feature_schema(w_pair=0.25)
    schema.update(version='sirna_engineered_v4_flank_composition', width=WIDTH,
                  w_flank=float(w_flank))
    offset = base.WIDTH
    for flank, positions in [('left',list(range(19))), ('right',list(range(38,57)))]:
        for k in (1,2,3):
            schema['blocks'].append(dict(name=f'{flank}_flank_{k}mer_counts',
                start=offset, stop=offset+4**k, positions=positions, channels=KMERS[k],
                divisor=float(np.sqrt(20-k)), scale=float(np.sqrt(w_flank)),
                windows='Overlapping within flank; exclude any window containing X; fixed potential-window denominator'))
            offset += 4**k
    return schema

def encode(frame, w_flank=0.25):
    feature_schema(w_flank)
    first = base.encode(frame, w_pair=0.25)
    out = np.zeros((len(frame), WIDTH), dtype=np.float64)
    out[:, :base.WIDTH] = first
    for row, context in enumerate(frame.extended_mRNA):
        offset = base.WIDTH
        for flank in (context[:19], context[38:57]):
            for k in (1,2,3):
                index = {word:i for i,word in enumerate(KMERS[k])}
                for start in range(20-k):
                    word = flank[start:start+k]
                    if 'X' not in word:
                        out[row,offset+index[word]] += 1
                out[row,offset:offset+4**k] /= np.sqrt(20-k)
                out[row,offset:offset+4**k] *= np.sqrt(w_flank)
                offset += 4**k
    return out
