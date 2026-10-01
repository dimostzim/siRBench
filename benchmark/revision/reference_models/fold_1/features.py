"""Fixed, stateless positional features. IDs and labels never enter transform."""
import numpy as np

GUIDE_ALPHABET = 'ACGT'
CONTEXT_ALPHABET = 'ACGTX'
FEATURE_NAMES = tuple(
    [f'guide_{p:02d}_{b}' for p in range(19) for b in GUIDE_ALPHABET]
    + [f'context_{p:02d}_{b}' for p in range(57) for b in CONTEXT_ALPHABET]
)
SCHEMA = {'version': 1, 'dtype': 'float32', 'n_features': 361,
          'guide_alphabet': GUIDE_ALPHABET, 'context_alphabet': CONTEXT_ALPHABET,
          'position_indexing': 'zero-based', 'feature_names': FEATURE_NAMES,
          'label_transform': 'identity', 'scaling': 'none'}


def transform(sequences):
    """Accept a DataFrame containing the two sequences; return (n,361).

    No fitted state, imputation, normalization, or cross-record interactions.
    Reject malformed inputs rather than silently substituting observed bases.
    """
    required = {'siRNA', 'extended_mRNA'}
    if not required.issubset(sequences.columns):
        raise ValueError('Missing sequence columns')
    if not sequences.columns.is_unique:
        raise ValueError('Duplicate column names')
    result = np.zeros((len(sequences), 361), dtype=np.float32)
    complement = str.maketrans('ACGT', 'TGCA')
    for row, (guide, context) in enumerate(
            sequences[['siRNA', 'extended_mRNA']].itertuples(index=False, name=None)):
        if not isinstance(guide, str) or len(guide) != 19 or set(guide) - set('ACGU'):
            raise ValueError(f'Invalid siRNA at input row {row}')
        if not isinstance(context, str) or len(context) != 57 or set(context) - set(CONTEXT_ALPHABET):
            raise ValueError(f'Invalid extended_mRNA at input row {row}')
        guide = guide.replace('U', 'T')
        if context[19:38] != guide.translate(complement)[::-1]:
            raise ValueError(f'Guide/context central-site mismatch at input row {row}')
        for p, base in enumerate(guide):
            result[row, 4 * p + GUIDE_ALPHABET.index(base)] = 1.0
        for p, base in enumerate(context):
            result[row, 76 + 5 * p + CONTEXT_ALPHABET.index(base)] = 1.0
    return result

# Preserve the preceding positional encoder verbatim, then append fixed blocks.
from itertools import product, groupby
_positional_transform = transform
POSITIONAL_NAMES = FEATURE_NAMES
KMERS = {k: tuple(map(''.join, product('ACGT', repeat=k))) for k in (1, 2, 3)}
REGIONS = ('guide', 'left', 'right')
FEATURE_NAMES = tuple(list(POSITIONAL_NAMES)
    + [f'guide_dimer_{p:02d}_{s}' for p in range(18) for s in KMERS[2]]
    + [f'{r}_k{k}_freq_{s}' for r in REGIONS for k in (1,2,3) for s in KMERS[k]]
    + [f'{r}_k{k}_coverage' for r in REGIONS for k in (1,2,3)]
    + [f'guide_gc_w{w}_p{p:02d}' for w in (3,5,7) for p in range(20-w)]
    + ['guide_gc', 'guide_gc_squared', 'guide_first5_gc', 'guide_last5_gc', 'guide_first5_minus_last5_gc']
    + [f'guide_max_run_{b}' for b in 'ACGT'])
SCHEMA = dict(SCHEMA, version=2, n_features=964, feature_names=FEATURE_NAMES,
    columns=[{'index': i, 'name': n} for i,n in enumerate(FEATURE_NAMES)],
    regions={'guide':'siRNA U->T, unchanged orientation', 'left':'context[0:19]', 'right':'context[38:57]'},
    blocks=[{'start':0,'stop':361,'rule':'Original guide 19x4 and context 57x5 positional one-hot; X explicit'},
            {'start':361,'stop':649,'rule':'Guide adjacent dimer positional one-hot; position-major, lexicographic ACGT dimers'},
            {'start':649,'stop':901,'rule':'Overlapping k-mer counts / number of valid windows, or zero if none; region-major, k=1,2,3, lexicographic ACGT words. X-containing windows excluded, never bridged'},
            {'start':901,'stop':910,'rule':'Valid window count / (19-k+1); region-major then k=1,2,3'},
            {'start':910,'stop':955,'rule':'GC count / window length; lengths 3,5,7 then ascending start position'},
            {'start':955,'stop':960,'rule':'Guide GC/19, its square, first-five GC/5, last-five GC/5, first-five fraction minus last-five fraction'},
            {'start':960,'stop':964,'rule':'Maximum contiguous run length for A,C,G,T separately; absent base zero'}],
    learned_preprocessing=False, exclusions=['IDs','row order','labels','annotations','external resources'])


def transform(sequences):
    """Return stateless (n,964) float32 numeric features, with no label access."""
    positional = _positional_transform(sequences)
    result = np.zeros((len(sequences), 964), dtype=np.float32)
    result[:, :361] = positional
    for i, (g,c) in enumerate(sequences[['siRNA','extended_mRNA']].itertuples(index=False, name=None)):
        g = g.replace('U','T')
        values = [float(g[p:p+2] == word) for p in range(18) for word in KMERS[2]]
        coverages = []
        for region in (g,c[:19],c[38:57]):
            for k in (1,2,3):
                counts = dict.fromkeys(KMERS[k], 0)
                for p in range(len(region)-k+1):
                    word = region[p:p+k]
                    if word in counts:
                        counts[word] += 1
                valid = sum(counts.values())
                values.extend(counts[word]/valid if valid else 0.0 for word in KMERS[k])
                coverages.append(valid/(len(region)-k+1))
        values.extend(coverages)
        gc = lambda s: sum(b in 'GC' for b in s)/len(s)
        values.extend(gc(g[p:p+w]) for w in (3,5,7) for p in range(20-w))
        values.extend([gc(g),gc(g)**2,gc(g[:5]),gc(g[-5:]),gc(g[:5])-gc(g[-5:])])
        runs = [(b,len(list(run))) for b,run in groupby(g)]
        values.extend(max((n for base,n in runs if base==b),default=0) for b in 'ACGT')
        result[i,361:] = values
    return result
