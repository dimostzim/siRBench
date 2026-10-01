#!/usr/bin/env python3
"""Return the completed temporary structure archive and validate it on node4."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np
import pandas as pd

root = Path('/SCRATCH/dtzim01/sirbench-revision-20260921')
stage = Path('/SCRATCH/dtzim01/sirbench-revision-20260921-attsioff-staging')
relative = Path('siRBench/benchmark/competitors/runs/_assets/sensitivity-v1/sirnadiscovery_full_mrna/original')
remote_asset, asset = stage / relative, root / relative
ssh = ['ssh', '-S', str(root / 'setup/ssh-node6/control'), '-o', 'BatchMode=yes', '10.200.29.6']
while True:
    command = ['test', '-f', remote_asset / 'siRNA_split_preprocess/complete.json']
    status = subprocess.run(ssh + [shlex.join(map(str, command))])
    if status.returncode == 0:
        break
    if status.returncode != 1:
        raise RuntimeError('Remote completion check failed')
    running = subprocess.check_output(ssh + ['docker ps --format {{.Names}}'], text=True).splitlines()
    if 'discovery-structure-full' not in running:
        raise RuntimeError('Structure worker stopped without completing the feature archive')
    time.sleep(30)
command = ['python3', '-c',
           'from pathlib import Path; import hashlib,sys; p=Path(sys.argv[1]); '
           'files=sorted(f for f in p.rglob("*") if f.is_file() and f.name!="SHA256SUMS"); '
           '(p/"SHA256SUMS").write_text("".join(hashlib.sha256(f.read_bytes()).hexdigest()+"  "+str(f.relative_to(p))+chr(10) for f in files))', remote_asset]
subprocess.run(ssh + [shlex.join(map(str, command))], check=True)
subprocess.run(['rsync', '-a', '-e', shlex.join(ssh[:-1]), f'10.200.29.6:{remote_asset}/', str(asset) + '/'], check=True)
for line in (asset / 'SHA256SUMS').read_text().splitlines():
    expected, filename = line.split('  ', 1)
    if hashlib.sha256((asset / filename).read_bytes()).hexdigest() != expected:
        raise ValueError(f'Archive transfer checksum mismatch: {filename}')
data = pd.read_csv(asset / 'all.csv')
expected = {'con_matrix.txt': (data.siRNA + '_' + data.mRNA, 50),
            'self_siRNA_matrix.txt': (data.siRNA, 6), 'self_mRNA_matrix.txt': (data.mRNA, 100)}
summary = {}
for name, (identifiers, components) in expected.items():
    features = pd.read_csv(asset / 'siRNA_split_preprocess' / name, header=None, index_col=0)
    if not features.index.is_unique or set(features.index) != set(identifiers):
        raise ValueError(f'Feature identifiers differ: {name}')
    if features.shape[1] != components or not np.isfinite(features.to_numpy()).all():
        raise ValueError(f'Invalid feature values: {name}')
    summary[name] = {'rows': len(features), 'components': components}
summary['all_sha256_verified'] = True
summary['asset_path'] = str(asset)
(root / 'audit/sirnadiscovery/full_rna_asset_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
print('VERIFIED_AND_RETURNED', json.dumps(summary), flush=True)
