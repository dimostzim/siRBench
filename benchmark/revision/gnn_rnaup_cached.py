"""Resumable parallel equivalent of GNN4siRNA's published 4_make_RNAUp.py.

Used only in isolated input-extent preprocessing. Input sequences and RNAup
arguments, including the published sense-strand choice, remain unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import pandas as pd

COMMAND = ['RNAup', '-b', '-d2', '--noLP', '-c', "'S'", 'RNAup.out']


def published_values(stdout):
    section = stdout[stdout.find(' (')+2:stdout.rfind(')\n')]
    values = list(map(float, section.replace('=', '').replace('+', '').split()))
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError('Expected four finite RNAup energy terms')
    return [values[0], values[2], values[3]]


def cached_interaction(task):
    guide_id, payload, cache, executable_hash = task
    specification = {'input': payload, 'command': COMMAND, 'executable_sha256': executable_hash}
    key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    path = Path(cache)/f'{key}.json'
    if path.exists():
        entry = json.loads(path.read_text())
        if entry['specification'] != specification or entry['values'] != published_values(entry['stdout']):
            raise ValueError(f'Invalid cached RNAup entry: {path}')
    else:
        with tempfile.TemporaryDirectory(prefix='rnaup-', dir=cache) as directory:
            result = subprocess.run(COMMAND, input=payload, text=True,
                                    capture_output=True, cwd=directory, check=True)
        entry = {'specification': specification, 'stdout': result.stdout,
                 'stderr': result.stderr, 'values': published_values(result.stdout)}
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(entry, indent=2)+'\n')
        temporary.replace(path)
    return [guide_id, *entry['values']]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)
    executable_hash = hashlib.sha256(Path(shutil.which('RNAup')).read_bytes()).hexdigest()
    stability = pd.read_csv('datase_tofold.csv', header=None)
    tasks = [(row[0], row[3]+'\n'+row[1], str(args.cache), executable_hash)
             for row in stability.itertuples(index=False, name=None)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(cached_interaction, tasks))
    pd.DataFrame(rows).to_csv('dataset_folded.csv')
    print(f'RNAup: {len(rows)} rows; content cache {args.cache}', flush=True)


if __name__ == '__main__':
    main()
