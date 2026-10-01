"""Select unfitted sequence features by verified IDs, never by row position."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_node_features(path):
    # Published preprocessing emits one target vector per interaction. Shared
    # full-length targets therefore repeat; only identical repeats are safe.
    features = pd.read_csv(path, header=None).drop_duplicates().set_index(0)
    if not features.index.is_unique:
        raise ValueError('Conflicting canonical features for the same node ID')
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError('Nonfinite node features')
    return features


def subset_features(source, records, destination, name):
    canonical = pd.read_csv(source/'all.csv').set_index('id')
    if not canonical.index.is_unique or not records.record_id.is_unique:
        raise ValueError('Canonical and requested record IDs must be unique')
    if not set(records.record_id).issubset(canonical.index):
        raise ValueError('Requested record missing from canonical features')
    subset = canonical.loc[records.record_id].copy()
    for requested, cached in [('siRNA', 'siRNA_seq'), ('extended_mRNA', 'mRNA_seq')]:
        sequences = records[requested].str.upper().str.replace('U', 'T')
        if sequences.tolist() != subset[cached].tolist():
            raise ValueError('Cached sequence differs from requested input')
    subset['efficiency'] = records.efficiency.to_numpy()
    directory = destination/'processed'/name
    directory.mkdir(parents=True, exist_ok=True)
    paths = [source/'all.csv']
    for filename, node_column in [('sirna_kmers.txt','siRNA'), ('target_kmers.txt','mRNA')]:
        path = source/'processed/all'/filename
        features = read_node_features(path)
        selected = features.loc[subset[node_column].drop_duplicates()]
        if not np.isfinite(selected.to_numpy()).all():
            raise ValueError('Nonfinite node features')
        selected.to_csv(directory/filename, header=False)
        paths.append(path)
    path = source/'processed/all/sirna_target_thermo.csv'
    interactions = pd.read_csv(path, header=None).set_index([0,1])
    if not interactions.index.is_unique:
        raise ValueError('Canonical interaction features contain duplicate IDs')
    keys = pd.MultiIndex.from_frame(subset[['siRNA','mRNA']])
    selected = interactions.loc[keys]
    if not np.isfinite(selected.to_numpy()).all():
        raise ValueError('Nonfinite interaction features')
    selected.to_csv(directory/path.name, header=False)
    paths.append(path)
    # Publish the prepared CSV last, after all required feature files are complete.
    subset.to_csv(destination/f'{name}.csv', index_label='id')
    manifest = {'n':len(subset),'rule':'Unfitted per-sequence/interaction features selected by ID after exact sequence verification; labels injected from this requested cohort.',
                'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                'output_sha256':{str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [destination/f'{name}.csv',*directory.glob('*')]}}
    (destination/f'{name}_feature_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--name',required=True)
    args = parser.parse_args()
    subset_features(args.source,pd.read_csv(args.input),args.output,args.name)


if __name__ == '__main__':
    main()
