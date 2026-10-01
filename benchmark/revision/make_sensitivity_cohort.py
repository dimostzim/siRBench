"""Freeze paired standardized/original-extent inputs on identical fold memberships."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def paired_inputs(partition, eligible):
    if not partition.record_id.is_unique or not eligible.record_id.is_unique:
        raise ValueError('Record IDs must be unique')
    standardized = partition[partition.record_id.isin(eligible.record_id)].copy()
    if standardized.empty:
        raise ValueError('No eligible records in partition')
    extents = eligible.set_index('record_id').loc[standardized.record_id]
    if not np.allclose(standardized.efficiency, extents.efficiency, rtol=0, atol=1e-12):
        raise ValueError('Extent labels differ from frozen protocol')
    if standardized.extended_mRNA.tolist() != extents.standardized_context_57.tolist():
        raise ValueError('Extent cores differ from frozen protocol')
    original_guides = extents.get('standardized_guide_19', extents.siRNA)
    if standardized.siRNA.tolist() != original_guides.tolist():
        raise ValueError('Guide cores differ from frozen protocol')
    original = standardized.copy()
    original['siRNA'] = extents.siRNA.to_numpy()
    original['extended_mRNA'] = extents.extended_mRNA.to_numpy()
    return standardized, original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--eligible', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-context-length', type=int)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new, empty sensitivity directory')
    eligible = pd.read_csv(args.eligible)
    exclusions = eligible.iloc[:0][['record_id']].copy()
    if args.max_context_length is not None:
        within_limit = eligible.extended_mRNA.str.len().le(args.max_context_length)
        exclusions = eligible.loc[~within_limit, ['record_id']].copy()
        exclusions['context_length'] = eligible.loc[~within_limit, 'extended_mRNA'].str.len()
        exclusions['reason'] = f'exceeds_original_maximum_{args.max_context_length}'
        eligible = eligible[within_limit]
    paths = {part: args.protocol/'grouped/fold_0'/f'{part}.csv'
             for part in ['train', 'val', 'test']}
    paths['hela_full'] = args.protocol/'hela_full.csv'
    inputs = {part: pd.read_csv(path) for part, path in paths.items()}
    all_ids = pd.concat([frame.record_id for frame in inputs.values()])
    if not all_ids.is_unique or not set(eligible.record_id).issubset(set(all_ids)):
        raise ValueError('Extent IDs must belong to the disjoint frozen partitions')
    paired = {part: paired_inputs(frame, eligible) for part, frame in inputs.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    sizes, outputs = [], []
    for part, frames in paired.items():
        sizes.append({'part': part, 'n': len(frames[0]), 'original_n': len(inputs[part])})
        for variant, frame in zip(['standardized', 'original'], frames):
            path = args.output/variant/'fold_0'/f'{part}.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            outputs.append(path)
    exclusions.to_csv(args.output/'length_exclusions.csv', index=False)
    pd.DataFrame(sizes).to_csv(args.output/'partition_sizes.csv', index=False)
    pd.DataFrame([{'variant': variant, 'axis': 'grouped', 'fold': 0, 'seed': seed}
                  for variant in ['standardized', 'original'] for seed in [0, 1, 2]]
                 ).to_csv(args.output/'run_matrix.csv', index=False)
    manifest = {
        'protocol': 'Matched input-extent sensitivity; grouped fold 0; common selection policy; seeds 0,1,2; identical training, validation and evaluation IDs and labels in both variants.',
        'selection': 'Fold 0 selected by index. This generator preserves the supplied eligibility list and frozen memberships without label-distribution or model-performance optimization. Eligibility criteria are documented separately in the source manifest; some provenance checks use agreement of normalized labels.',
        'features': 'Tools must prepare features from their variant-specific sequence inputs. Copied benchmark 57-nt feature columns are provenance only, not original-extent features.',
        'max_context_length': args.max_context_length, 'partition_sizes': sizes,
        'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [args.eligible, *paths.values(), Path(__file__)]},
        'output_sha256': {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in outputs},
    }
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(sizes))


if __name__ == '__main__':
    main()
