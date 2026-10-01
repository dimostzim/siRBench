"""Wait for the six primary matrices, then publish one complete analysis."""
import argparse
import hashlib
import itertools
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from assemble_primary_index import KEY, TOOLS, complete_index, completed_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace
    indexes = [root/'evaluation/comparator_runs'/name for name in [
        'oligo_bert_run_index.csv', 'oligo_bert_random_run_index.csv',
        'attsioff_run_index.csv', 'sirnadiscovery_run_index.csv']]
    indexes += [root/'evaluation/ensirna-run-index.csv',
                root/'siRBench/benchmark/competitors/runs/main-gnn-v1/run_index.csv']
    expected = set(itertools.product(TOOLS, ['grouped', 'random'], range(5), range(3)))
    previous = None
    while True:
        paths = [path for path in indexes if path.is_file()]
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        frames = [pd.read_csv(path) for path in paths]
        after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        if before != after:
            time.sleep(2)
            continue
        if frames:
            index = pd.concat(frames, ignore_index=True)
            if index.duplicated(KEY).any() or not set(map(tuple, index[KEY].to_numpy())).issubset(expected):
                raise ValueError('Duplicate or non-primary runs in the supplied indexes')
            verified = index.loc[completed_runs(index)]
            counts = verified.groupby('tool').size().to_dict()
            if counts != previous:
                print(time.strftime('%Y-%m-%d %H:%M:%S'), counts, flush=True)
                previous = counts
            if len(verified) == len(expected):
                complete_index(frames)
                break
        time.sleep(60)
    code = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(code/'assemble_primary_index.py'),
                    *[item for path in indexes for item in ['--run-index', str(path)]],
                    '--output', str(root/'evaluation/primary-index-v1')], check=True)
    subprocess.run([sys.executable, str(code/'evaluate_predictions.py'),
                    '--records', str(root/'datasets/corrected-v1/records_features.csv'),
                    '--protocol', str(root/'evaluation/protocol-v1'),
                    '--baselines', str(root/'evaluation/baselines-v1/predictions.csv'),
                    '--run-index', str(root/'evaluation/primary-index-v1/run_index.csv'),
                    '--output', str(root/'evaluation/primary-analysis-v1'),
                    '--bootstrap-draws', '2000'], check=True)
    print('Complete primary analysis written and input hashes recorded.', flush=True)


if __name__ == '__main__':
    main()
