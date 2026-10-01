"""Fixed, stateless positional encoding; only NumPy is required."""
import csv
from pathlib import Path
import numpy as np

CONFIG = {
    'version': 'positional_onehot_v1',
    'guide_length': 19, 'guide_alphabet': 'ACGT',
    'context_length': 57, 'context_alphabet': 'ACGTX',
    'guide_conversion': 'U->T; no reversal or complementation',
    'ordering': 'guide then context; position-major, alphabet-minor',
    'dtype': 'float64', 'n_features': 361,
    'label_transform': 'identity', 'learned_preprocessing': False,
}

def positional_feature_names():
    return ([f'guide_{p:02d}_{b}' for p in range(19) for b in 'ACGT'] +
            [f'context_{p:02d}_{b}' for p in range(57) for b in 'ACGTX'])

def encode_positional(guides, contexts):
    """Encode ordered sequence pairs. Reject invalid input; never drop records."""
    guides, contexts = list(guides), list(contexts)
    if len(guides) != len(contexts):
        raise ValueError('Guide/context count mismatch')
    features = np.zeros((len(guides), 361), dtype=np.float64)
    for i, (guide, context) in enumerate(zip(guides, contexts)):
        if not isinstance(guide, str) or len(guide) != 19 or set(guide) - set('ACGU'):
            raise ValueError(f'Invalid 19-nt ACGU guide at row {i}')
        if not isinstance(context, str) or len(context) != 57 or set(context) - set('ACGTX'):
            raise ValueError(f'Invalid 57-position ACGTX context at row {i}')
        guide = guide.replace('U', 'T')
        for p, base in enumerate(guide):
            features[i, 4*p + 'ACGT'.index(base)] = 1.0
        for p, base in enumerate(context):
            features[i, 76 + 5*p + 'ACGTX'.index(base)] = 1.0
    return features

def load_input(input_dir):
    """Read only <input_dir>/data.csv; return output keys separately from features."""
    with (Path(input_dir) / 'data.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        if not {'id', 'siRNA', 'extended_mRNA'}.issubset(reader.fieldnames or []):
            raise ValueError('Missing required input columns')
        rows = list(reader)
    ids = [r['id'] for r in rows]
    if any(not k for k in ids) or len(set(ids)) != len(ids):
        raise ValueError('Empty or duplicate IDs')
    return ids, encode_sequences([r['siRNA'] for r in rows], [r['extended_mRNA'] for r in rows])

def load_labels(labels_path, ids):
    """Training/evaluation only: one-to-one ID alignment, identity label transform."""
    with Path(labels_path).open(newline='') as f:
        reader = csv.DictReader(f)
        column = 'numeric_label' if 'numeric_label' in (reader.fieldnames or []) else 'label'
        rows = list(reader)
    mapping = {r['id']: float(r[column]) for r in rows}
    if len(mapping) != len(rows) or len(set(ids)) != len(ids) or set(mapping) != set(ids):
        raise ValueError('Labels must match input IDs one-to-one')
    y = np.asarray([mapping[k] for k in ids], dtype=np.float64)
    if not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
        raise ValueError('Expected finite efficacy labels in [0,1]')
    return y

from itertools import product, groupby
import math

GC_WINDOWS = [(0,19),(0,4),(0,7),(6,13),(12,19),(15,19),(1,8)]

def engineered(guide, context):
    """Return ordered (name, value, definition) triples; windows are overlapping."""
    out = []
    def add(name, value, definition):
        out.append((name, float(value), definition))
    def kmers(seq, region, ks, padding=False):
        for k in ks:
            windows = [seq[i:i+k] for i in range(len(seq)-k+1)]
            valid = [w for w in windows if 'X' not in w]
            for letters in product('ACGT', repeat=k):
                word = ''.join(letters)
                add(f'{region}_k{k}_{word}', valid.count(word)/len(valid) if valid else 0,
                    f'Overlapping {word} count in {region} / number of X-free length-{k} windows; 0 if none')
            if padding:
                add(f'{region}_k{k}_valid_fraction', len(valid)/len(windows),
                    f'X-free length-{k} windows in {region} / {len(windows)}')
    kmers(guide, 'guide', (1,2,3))
    kmers(context[:19], 'left', (1,2), True)
    kmers(context[38:], 'right', (1,2), True)
    def gc(seq):
        observed = sum(seq.count(b) for b in 'ACGT')
        return (seq.count('G')+seq.count('C'))/observed if observed else 0
    for a,b in GC_WINDOWS:
        add(f'guide_gc_{a}_{b}', gc(guide[a:b]), f'GC count / {b-a} in guide[{a}:{b}], zero-based half-open')
    add('guide_gc_end_difference',gc(guide[:4])-gc(guide[-4:]),'GC fraction guide[:4] minus guide[15:19]')
    runs = [(b,len(list(g))) for b,g in groupby(guide)]
    for base in 'ACGT':
        add(f'guide_run_{base}',max([n for b,n in runs if b==base], default=0)/19, f'Longest contiguous {base} run / 19; 0 if absent')
    add('guide_run_any',max(n for b,n in runs)/19,'Longest same-base run / 19')
    p = [guide.count(b)/19 for b in 'ACGT']
    add('guide_entropy',-sum(x*math.log2(x) for x in p if x),'Mononucleotide Shannon entropy in bits; omit zero terms')
    for region,seq in [('left',context[:19]),('right',context[38:])]:
        add(f'{region}_observed_gc',gc(seq),f'GC count / ACGT count in {region}; 0 if no observed bases')
        add(f'{region}_x_fraction',seq.count('X')/19,f'X count in {region} / 19')
    add('context_x_fraction',context.count('X')/57,'Full context X count / 57')
    add('flank_gc_difference',gc(context[:19])-gc(context[38:]),'Left minus right observed-base GC fractions; empty flank GC is 0')
    for motif in ['AAAA','CCCC','GGGG','TTTT']:
        add(f'guide_motif_{motif}',sum(guide[i:i+4]==motif for i in range(16))/16,f'Overlapping {motif} count in guide / 16')
    return out

_TEMPLATE = engineered('A'*19, 'X'*57)
PAIRS = [''.join(p) for p in product('ACGT', repeat=2)]
CONFIG.update(version='positional_composition_dinucleotide_v3', n_features=801,
              ordering='513 unchanged positional/composition columns, then 18 position-major blocks of 16 ACGT-lexicographic guide dinucleotide indicators',
              zero_denominator='All observed-base and valid-window ratios return 0 when denominator is 0')

def feature_names():
    return (positional_feature_names()+[name for name,_,_ in _TEMPLATE]+
            [f'guide_dinucleotide_{p:02d}_{pair}' for p in range(18) for pair in PAIRS])

def representation_manifest():
    definitions = ([f'Indicator guide[{p}] == {b} after U->T' for p in range(19) for b in 'ACGT']+
                   [f'Indicator context[{p}] == {b}; X explicit' for p in range(57) for b in 'ACGTX']+
                   [definition for _,_,definition in _TEMPLATE]+
                   [f'Indicator guide[{p}:{p+2}] == {pair} after U->T; no reversal/complement' for p in range(18) for pair in PAIRS])
    return dict(CONFIG, features=[{'index':i,'name':name,'definition':definition}
                                 for i,(name,definition) in enumerate(zip(feature_names(),definitions))])

def encode_sequences(guides, contexts):
    guides, contexts = list(guides), list(contexts)
    positional = encode_positional(guides, contexts)
    extra = np.asarray([[v for _,v,_ in engineered(g.replace('U','T'),c)]
                        for g,c in zip(guides,contexts)],dtype=np.float64).reshape(len(guides),len(_TEMPLATE))
    dinucleotides = np.zeros((len(guides), 288), dtype=np.float64)
    for row, guide in enumerate(guides):
        guide = guide.replace('U', 'T')
        for p in range(18):
            dinucleotides[row, p*16 + PAIRS.index(guide[p:p+2])] = 1.0
    result = np.concatenate([positional,extra,dinucleotides],axis=1)
    assert result.shape == (len(guides),CONFIG['n_features']) and np.isfinite(result).all()
    return result
