"""Run one disjoint ten-fit worker queue, preserving failed attempts."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from run_validation import ROOT, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--attempt', type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    output = ROOT / 'evaluation/tabpfn-benchmark-v1'
    output.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).with_name('run_benchmark.py').resolve()
    plan = [(axis, fold, seed) for axis_number, axis in enumerate(['grouped', 'random'])
            for fold in range(5) for seed in range(3)
            if (axis_number + fold + seed) % 3 == args.worker]
    worker_path = output / f'worker_{args.worker}_attempt_{args.attempt}.json'
    if worker_path.exists():
        raise ValueError('Worker already launched; inspect previous attempt before restarting')
    manifest = {'host': socket.gethostname(), 'pid': os.getpid(), 'worker': args.worker, 'attempt': args.attempt,
                'plan': plan, 'wrapper_sha256': sha256(code), 'completed': [], 'failed': [],
                'started_unix': time.time(), 'status': 'running'}
    write_json(worker_path, manifest)
    for axis, fold, seed in plan:
        run_dir = output / 'runs' / axis / f'fold_{fold}' / f'seed_{seed}' / f'attempt_{args.attempt}'
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        log_path = run_dir.parent / f'attempt_{args.attempt}.log'
        with log_path.open('x') as handle:
            result = subprocess.run([sys.executable, '-u', str(code), '--axis', axis,
                                     '--fold', str(fold), '--seed', str(seed), '--output', str(run_dir)],
                                    stdout=handle, stderr=subprocess.STDOUT, check=False)
        record = {'axis': axis, 'fold': fold, 'seed': seed, 'returncode': result.returncode,
                  'output': str(run_dir), 'log': str(log_path)}
        manifest['completed' if result.returncode == 0 else 'failed'].append(record)
        write_json(worker_path, manifest)
        print(json.dumps(record), flush=True)
    manifest['status'] = 'complete' if not manifest['failed'] else 'needs_attention'
    manifest['finished_unix'] = time.time()
    write_json(worker_path, manifest)
    if manifest['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
