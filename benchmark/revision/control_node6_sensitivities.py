#!/usr/bin/env python3
"""Seal, execute, return and verify explicitly specified node6 sensitivity runs."""
import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import shlex
import subprocess
import time

from run_attsioff_discovery_matrix import remote, digest, check_model_completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--remote-root', type=Path, required=True)
    parser.add_argument('--host', default='10.200.29.6')
    parser.add_argument('--control-socket', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    repo, remote_repo = args.workspace / 'siRBench', args.remote_root / 'siRBench'
    index_path = args.workspace / 'evaluation/comparator_runs/node6_sensitivity_run_index.csv'
    index_lock = index_path.with_suffix('.lock').open('a')
    fcntl.flock(index_lock, fcntl.LOCK_EX)
    fields = ['tool', 'axis', 'fold', 'training_seed', 'run_dir', 'test_predictions', 'hela_predictions',
              'train_meta', 'status', 'elapsed_seconds']
    index = list(csv.DictReader(index_path.open())) if index_path.exists() else []
    completed = {(row['axis'], int(row['training_seed'])) for row in index if row['status'] == 'verified'}
    for specification in json.loads(args.matrix.read_text()):
        case, seed, tool = specification['case'], specification['seed'], specification['tool']
        if args.case and case not in args.case:
            continue
        if (case, seed) in completed:
            continue
        relative = Path(specification['run_relative'])
        central_run, remote_run = repo / relative, remote_repo / relative
        run_args = ['--tool', tool, '--run-dir', remote_run, '--seed', str(seed)]
        for role, path in specification['inputs'].items():
            run_args += ['--' + role, args.remote_root / path]
        print('START', case, seed, flush=True)
        started = time.time()
        remote(args, ['python3', remote_repo / 'benchmark/competitors/scripts/prepare_run.py',
                      '--repo-root', remote_repo, '--original', '0', '--deterministic', '0', *run_args])
        spec_name = f'{case}_seed{seed}.json'
        spec_path = repo / 'benchmark/revision/sensitivity_specs' / spec_name
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(specification, indent=2) + '\n')
        remote_spec_dir = remote_repo / 'benchmark/revision/sensitivity_specs'
        remote(args, ['mkdir', '-p', remote_spec_dir])
        ssh_args = shlex.join(['ssh', '-S', str(args.control_socket), '-o', 'BatchMode=yes'])
        subprocess.run(['rsync', '-a', '-e', ssh_args, str(spec_path), f'{args.host}:{remote_spec_dir}/'], check=True)
        with (args.workspace / 'setup' / f'sensitivity-{case}-seed{seed}.log').open('a') as log:
            remote(args, ['python3', remote_repo / 'benchmark/revision/run_node6_sensitivity.py', '--repo', remote_repo,
                          '--specification', remote_spec_dir / spec_name], stdout=log, stderr=subprocess.STDOUT)
        remote(args, ['python3', '-c',
            'from pathlib import Path; import hashlib,sys; p=Path(sys.argv[1]); '
            'files=sorted(f for f in p.rglob("*") if f.is_file() and f.name!="SHA256SUMS"); '
            '(p/"SHA256SUMS").write_text("".join(hashlib.sha256(f.read_bytes()).hexdigest()+"  "+str(f.relative_to(p))+chr(10) for f in files))', remote_run])
        central_run.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync', '-a', '-e', ssh_args, f'{args.host}:{remote_run}/', str(central_run) + '/'], check=True)
        for line in (central_run / 'SHA256SUMS').read_text().splitlines():
            expected, filename = line.split('  ', 1)
            if digest(central_run / filename) != expected:
                raise ValueError(f'Transfer checksum mismatch: {filename}')
        for filename, expected in json.loads((central_run / 'SENSITIVITY_COMPLETE.json').read_text()).items():
            if digest(central_run / filename) != expected:
                raise ValueError(f'Completion checksum mismatch: {filename}')
        check_model_completion(central_run, tool, seed, required=True)
        predictions = {}
        for role, name in [('test', 'preds.csv'), ('leftout', 'preds_leftout.csv')]:
            path = central_run / 'results' / tool / name
            inputs = list(csv.DictReader((central_run / 'inputs' / (role + '.csv')).open()))
            preds = list(csv.DictReader(path.open()))
            if [row['record_id'] for row in inputs] != [row['id'] for row in preds]:
                raise ValueError(f'Prediction record IDs/order differ: {path}')
            if any(abs(float(row['efficiency']) - float(pred['label'])) > 1e-7 for row, pred in zip(inputs, preds)):
                raise ValueError(f'Prediction labels differ: {path}')
            if not all(math.isfinite(float(pred['pred_label'])) for pred in preds):
                raise ValueError(f'Nonfinite prediction: {path}')
            predictions[role] = str(path)
        index.append(dict(zip(fields, [tool, case, 0, seed, str(central_run), predictions['test'], predictions['leftout'],
                                     str(central_run / 'models' / tool / 'train_meta.json'), 'verified', round(time.time() - started, 1)])))
        with index_path.with_suffix('.tmp').open('w') as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(index)
        index_path.with_suffix('.tmp').replace(index_path)
        remote(args, ['python3', '-c', 'from pathlib import Path; import shutil,sys; p,root=map(Path,sys.argv[1:]); '
                      'p.resolve().relative_to(root.resolve()); shutil.rmtree(p)', remote_run,
                      remote_repo / 'benchmark/competitors/runs/sensitivity_v1'])
        print('VERIFIED_AND_RETURNED', case, seed, round(time.time() - started, 1), flush=True)


if __name__ == '__main__':
    main()
