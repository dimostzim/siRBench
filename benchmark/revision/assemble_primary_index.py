"""Publish an index only when all six methods have complete verified matrices."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

from prediction_artifacts import ARTIFACT_COLUMNS, read_artifact_bytes, validate_artifact_paths

TOOLS = {'oligoformer','sirnabert','attsioff','sirnadiscovery','gnn4sirna','ensirna'}
KEY = ['tool','axis','fold','training_seed']


def completed_runs(index):
    # EN's central collector calls its fully validated/returned state complete.
    # Worker indexes must not be supplied directly to primary assembly.
    return index.status.eq('verified') | (index.tool.eq('ensirna') & index.status.eq('complete'))


def complete_index(frames):
    index = pd.concat(frames,ignore_index=True)
    expected = set(itertools.product(TOOLS,['grouped','random'],range(5),range(3)))
    if index.duplicated(KEY).any():
        raise ValueError('Duplicate primary run across supplied indexes')
    if set(map(tuple,index[KEY].to_numpy())) != expected or not completed_runs(index).all():
        raise ValueError('Expected all180 verified primary runs; partial/sensitivity indexes are not accepted')
    validate_artifact_paths(index)
    for column in ARTIFACT_COLUMNS:
        index[column + '_sha256'] = [read_artifact_bytes(row, column)[1]
                                    for row in index.itertuples(index=False)]
    index['source_status'] = index.status
    index['status'] = 'verified'
    return index.sort_values(KEY).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-index',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new, empty index output directory')
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.run_index}
    index = complete_index([pd.read_csv(p) for p in args.run_index])
    if hashes != {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.run_index}:
        raise ValueError('An input index changed during collection; retry once stable')
    args.output.mkdir(parents=True,exist_ok=True)
    index.to_csv(args.output/'run_index.csv',index=False)
    manifest = {'runs':len(index),'tools':sorted(TOOLS),'source_indexes':hashes,
                'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'artifact_validation_sha256':hashlib.sha256(Path(__file__).with_name('prediction_artifacts.py').read_bytes()).hexdigest(),
                'index_sha256':hashlib.sha256((args.output/'run_index.csv').read_bytes()).hexdigest()}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
