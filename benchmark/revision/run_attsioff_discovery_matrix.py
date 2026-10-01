"""Run the frozen AttSiOff/siRNADiscovery matrix on temporary node6 staging.

Execute on node4 through an agent-forwarded SSH session. Each successful run is
copied, SHA256-verified, indexed centrally, then removed from node6. Per-sequence RNA-FM and RNAfold/RPISeq features are reused from a frozen archive.
Every run receives its own linked files and an asset checksum manifest.
"""
import argparse
import csv
import hashlib
import math
import json
from pathlib import Path
import shlex
import subprocess
import time


def remote(args, arguments, **kwargs):
    return subprocess.run(['ssh', '-S', str(args.control_socket), '-o', 'BatchMode=yes', args.host, shlex.join([str(x) for x in arguments])], check=True, **kwargs)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()



def check_model_completion(run_dir, tool, seed, required=False):
    model_dir = Path(run_dir) / "models" / tool
    meta_path = model_dir / "train_meta.json"
    if (required or any(model_dir.glob("model.*"))) and not meta_path.is_file():
        raise ValueError("Checkpoint lacks training completion metadata; do not resume this run")
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text())
        if metadata.get("seed") != int(seed):
            raise ValueError("Training metadata seed differs from the requested run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--remote-root', type=Path, required=True)
    parser.add_argument('--host', default='10.200.29.6')
    parser.add_argument('--control-socket', type=Path, required=True)
    parser.add_argument('--tool', choices=['attsioff', 'sirnadiscovery'], action='append', required=True)
    args = parser.parse_args()
    repo = args.workspace / 'siRBench'
    remote_repo = args.remote_root / 'siRBench'
    protocol = args.workspace / 'evaluation/protocol-v1'
    index_dir = args.workspace / 'evaluation/comparator_runs'
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / ('_'.join(args.tool) + '_run_index.csv')
    fields = ['tool', 'axis', 'fold', 'training_seed', 'run_dir', 'test_predictions',
              'hela_predictions', 'train_meta', 'status', 'elapsed_seconds']
    index = list(csv.DictReader(index_path.open())) if index_path.exists() else []
    completed = {(r['tool'], r['axis'], r['fold'], r['training_seed']) for r in index if r['status'] == 'verified'}
    for row in csv.DictReader((protocol / 'run_matrix.csv').open()):
        for tool in args.tool:
            key = (tool, row['axis'], row['fold'], row['training_seed'])
            if key in completed:
                continue
            run_name = f"{row['axis']}_fold{row['fold']}_seed{row['training_seed']}"
            relative = Path('benchmark/competitors/runs/revision_v1') / tool / run_name
            central_run = repo / relative
            remote_run = remote_repo / relative
            assets = remote_repo / 'benchmark/competitors/runs/_assets' / tool
            paths = {role: args.remote_root / Path(row[column]).relative_to(args.workspace)
                     for role, column in [('train','train'), ('val','val'), ('test','test'), ('leftout','hela_full')]}
            run_args = ['--tool', tool, '--run-dir', remote_run, '--seed', row['training_seed']]
            for role, path in paths.items():
                run_args += ['--' + role, path]
            seal = ['python3', remote_repo / 'benchmark/competitors/scripts/prepare_run.py',
                    '--repo-root', remote_repo, '--original', '0', '--deterministic', '0', *run_args]
            print('START', tool, run_name, flush=True)
            started = time.time()
            remote(args, seal)
            remote(args, ['python3', '-c',
                'import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); '
                'from run_attsioff_discovery_matrix import check_model_completion; '
                'check_model_completion(Path(sys.argv[2]),sys.argv[3],sys.argv[4])',
                remote_repo/'benchmark/revision', remote_run, tool, row['training_seed']])
            remote(args, ['python3', '-c',
                'from pathlib import Path; import shutil,os,sys; a,d=map(Path,sys.argv[1:3]); tool=sys.argv[3]; '
                'd.mkdir(parents=True,exist_ok=True); '
                'pairs=[("data","data")] if tool=="attsioff" else '
                '[("RNA_AGO2","RNA_AGO2"),("siRNA_split_preprocess","siRNA_split_preprocess_trainval"),'
                '("siRNA_split_preprocess","siRNA_split_preprocess_test"),("siRNA_split_preprocess","siRNA_split_preprocess_leftout")]; '
                '[(shutil.copytree(a/s,d/t,copy_function=os.link,symlinks=False) if not (d/t).exists() else None) for s,t in pairs]; '
                'shutil.copyfile(a/"SHA256SUMS",d/"ASSET_SHA256SUMS")', assets, remote_run/'data'/tool, tool])
            log_path = args.workspace / 'setup' / f'{tool}-{run_name}.log'
            with log_path.open('a') as log:
                remote(args, ['env', 'QUIET=0', 'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4',
                    'bash', remote_repo / 'benchmark/competitors/run_tool.sh', *run_args], stdout=log, stderr=subprocess.STDOUT)
            remote(args, ['python3', '-c',
                'from pathlib import Path; import hashlib,sys; p=Path(sys.argv[1]); '
                'files=sorted(f for f in p.rglob("*") if f.is_file() and f.name!="SHA256SUMS"); '
                '(p/"SHA256SUMS").write_text("".join(hashlib.sha256(f.read_bytes()).hexdigest()+"  "+str(f.relative_to(p))+chr(10) for f in files))', remote_run])
            central_run.mkdir(parents=True, exist_ok=True)
            subprocess.run(['rsync', '-a', '-e', shlex.join(['ssh', '-S', str(args.control_socket), '-o', 'BatchMode=yes']), f'{args.host}:{remote_run}/', str(central_run) + '/'], check=True)
            for line in (central_run / 'SHA256SUMS').read_text().splitlines():
                expected, filename = line.split('  ', 1)
                if digest(central_run / filename) != expected:
                    raise ValueError(f'Transfer checksum mismatch: {filename}')
            check_model_completion(central_run, tool, row["training_seed"], required=True)
            predictions = {}
            for role, filename in [('test','preds.csv'), ('leftout','preds_leftout.csv')]:
                path = central_run / 'results' / tool / filename
                inputs = list(csv.DictReader((central_run/'inputs'/f'{role}.csv').open()))
                preds = list(csv.DictReader(path.open()))
                if [r['record_id'] for r in inputs] != [r['id'] for r in preds]:
                    raise ValueError(f'Prediction record IDs/order differ: {path}')
                if any(abs(float(r['efficiency'])-float(p['label']))>1e-7 for r,p in zip(inputs,preds)):
                    raise ValueError(f'Prediction labels differ: {path}')
                if not all(math.isfinite(float(p['pred_label'])) for p in preds):
                    raise ValueError(f'Nonfinite prediction: {path}')
                predictions[role] = str(path)
            index.append(dict(zip(fields, [tool, row['axis'], row['fold'], row['training_seed'], str(central_run),
                predictions['test'], predictions['leftout'], str(central_run/'models'/tool/'train_meta.json'),
                'verified', str(round(time.time()-started, 1))])))
            with index_path.with_suffix('.tmp').open('w') as handle:
                writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(index)
            index_path.with_suffix('.tmp').replace(index_path)
            remote(args, ['python3', '-c',
                'from pathlib import Path; import shutil,sys; p,root=map(Path,sys.argv[1:]); '
                'p.resolve().relative_to(root.resolve()); shutil.rmtree(p)', remote_run, remote_repo/'benchmark/competitors/runs/revision_v1'])
            completed.add(key)
            print('VERIFIED_AND_RETURNED', tool, run_name, 'seconds', round(time.time()-started, 1), flush=True)


if __name__ == '__main__':
    main()
