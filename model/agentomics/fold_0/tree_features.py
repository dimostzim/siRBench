"""Fixed CatBoost view: ordered schema-based removal, no learned transforms."""
from copy import deepcopy
import numpy as np
import sequence_features as source

WIDTH = 1058

def feature_schema():
    original = source.feature_schema(w_flank=0.25)
    blocks, indices, offset, excluded = [], [], 0, 0
    for block in original['blocks']:
        start, stop = block['start'], block['stop']
        if block['name'] == 'guide_nonadjacent_pair_position':
            excluded += stop-start
            continue
        b = deepcopy(block)
        b.update(source_start=start, source_stop=stop, start=offset, stop=offset+stop-start)
        blocks.append(b)
        indices.extend(range(start,stop))
        offset += stop-start
    assert excluded == 2448 and offset == WIDTH and len(indices) == WIDTH
    return dict(version='sirna_tree_v1',dtype='float64',width=WIDTH,
                source_width=source.WIDTH, source_w_flank=0.25,
                excluded_block='guide_nonadjacent_pair_position',blocks=blocks,
                source_column_indices=indices, normalization='none; retain fixed source block scales',
                label_transform='none; original continuous efficacy')

INDICES = np.asarray(feature_schema()['source_column_indices'],dtype=np.int64)

def encode(frame):
    return source.encode(frame,w_flank=0.25)[:,INDICES]
