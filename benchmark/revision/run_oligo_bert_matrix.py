"""Run the frozen OligoFormer/BERT matrix on temporary node5 staging.

Execute on node4 through an agent-forwarded SSH session. Each successful run is
copied, SHA256-verified, indexed centrally, then removed from node5. OligoFormer
frozen RNA-FM preparation is reused only across seeds of the identical partition.
"""
import argparse
import csv
import hashlib
import math
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


CHECK_TRAINING_COMPLETE = """
import json, sys
from pathlib import Path
run, tool, seed, original, required = sys.argv[1:]
model = Path(run)/"models"/tool/"model.pt"
meta = model.parent/"train_meta.json"
if model.exists():
    if not meta.is_file():
        raise ValueError("Checkpoint exists without completed training metadata")
    settings = json.loads(meta.read_text())
    epochs, patience, metric = ((200,30,"loss+auc") if tool=="oligoformer" else (30,0,"val_loss")) if int(original) else (100,20,"r2" if tool=="oligoformer" else "val_r2")
    expected = {"seed":int(seed),"epochs":epochs,"early_stopping":patience,"early_stop_metric":metric}
    if any(settings.get(key)!=value for key,value in expected.items()) or settings.get("best_epoch",-1)<0:
        raise ValueError("Completed training metadata differs from requested settings")
    if int(original) and tool=="sirnabert":
        if settings.get("checkpoint_selection")!="final_epoch" or settings.get("best_epoch")!=29 or settings.get("epochs_completed")!=30:
            raise ValueError("Original BERT schedule requires the completed final epoch checkpoint")
elif int(required) or meta.exists():
    raise ValueError("Completed model checkpoint is missing")
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--remote-root', type=Path, required=True)
    parser.add_argument('--host', default='10.200.29.5')
    parser.add_argument('--control-socket', type=Path, required=True)
    parser.add_argument('--tool', choices=['oligoformer', 'sirnabert'], action='append', required=True)
    parser.add_argument('--axis', choices=['grouped', 'random'])
    parser.add_argument('--fold', type=int, choices=range(5))
    parser.add_argument('--index', type=Path)
    parser.add_argument('--original', action='store_true')
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    if args.original and not args.index:
        parser.error('--original requires a separate --index')
    repo = args.workspace / 'siRBench'
    remote_repo = args.remote_root / 'siRBench'
    protocol = args.workspace / 'evaluation/protocol-v1'
    index_dir = args.workspace / 'evaluation/comparator_runs'
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.index or index_dir / 'oligo_bert_run_index.csv'
    series = 'revision_v1_original' if args.original else 'revision_v1'
    fields = ['tool', 'axis', 'fold', 'training_seed', 'run_dir', 'test_predictions',
              'hela_predictions', 'train_meta', 'status', 'elapsed_seconds']
    index = list(csv.DictReader(index_path.open())) if index_path.exists() else []
    completed = {(r['tool'], r['axis'], r['fold'], r['training_seed']) for r in index if r['status'] == 'verified'}
    if len(completed) != len(index):
        raise ValueError('Index contains duplicate or unverified run identities')
    for row in csv.DictReader((protocol / 'run_matrix.csv').open()):
        if args.axis and row['axis'] != args.axis:
            continue
        if args.fold is not None and int(row['fold']) != args.fold:
            continue
        for tool in args.tool:
            if args.stop_file and args.stop_file.exists():
                print('STOPPED_BEFORE_NEXT_RUN', args.stop_file, flush=True)
                return
            key = (tool, row['axis'], row['fold'], row['training_seed'])
            if key in completed:
                continue
            run_name = f"{row['axis']}_fold{row['fold']}_seed{row['training_seed']}"
            relative = Path('benchmark/competitors/runs') / series / tool / run_name
            central_run = repo / relative
            remote_run = remote_repo / relative
            cache = args.remote_root / ('prepared_cache_original' if args.original else 'prepared_cache') / tool / f"{row['axis']}_fold{row['fold']}"
            paths = {role: args.remote_root / Path(row[column]).relative_to(args.workspace)
                     for role, column in [('train','train'), ('val','val'), ('test','test'), ('leftout','hela_full')]}
            run_args = ['--tool', tool, '--run-dir', remote_run, '--seed', row['training_seed']]
            for role, path in paths.items():
                run_args += ['--' + role, path]
            seal = ['python3', remote_repo / 'benchmark/competitors/scripts/prepare_run.py',
                    '--repo-root', remote_repo, '--original', str(int(args.original)), '--deterministic', '0', *run_args]
            print('START', tool, run_name, flush=True)
            started = time.time()
            remote(args, seal)
            completion_args = [remote_run, tool, row['training_seed'], str(int(args.original))]
            remote(args, ['python3', '-c', CHECK_TRAINING_COMPLETE, *completion_args, '0'])
            if tool == 'oligoformer':
                remote(args, ['python3', '-c',
                    'from pathlib import Path; import shutil,os,sys,json; c,d=map(Path,sys.argv[1:]); '
                    'a=json.loads((c/"_prepared_from_manifest.json").read_text()) if c.exists() else None; '
                    'b=json.loads((d.parent/"manifest.json").read_text()); '
                    'a["settings"].pop("seed") if a else None; b["settings"].pop("seed"); '
                    'assert a is None or a==b, "Prepared cache differs from current inputs/code/image"; '
                    'shutil.copytree(c,d,copy_function=os.link) if c.exists() and not d.exists() else None',
                    cache, remote_run / 'data'])
            log_path = args.workspace / 'setup' / f"{'original-' if args.original else ''}{tool}-{run_name}.log"
            with log_path.open('a') as log:
                remote(args, ['env', 'QUIET=0', 'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4',
                    'bash', remote_repo / 'benchmark/competitors/run_tool.sh', *run_args, *(['--original'] if args.original else [])], stdout=log, stderr=subprocess.STDOUT)
            remote(args, ['python3', '-c', CHECK_TRAINING_COMPLETE, *completion_args, '1'])
            # Copy complete preparation for later seeds before removing this run.
            if tool == 'oligoformer' and row['training_seed'] == '0':
                remote(args, ['python3', '-c',
                    'from pathlib import Path; import shutil,os,sys; s,d=map(Path,sys.argv[1:]); '
                    'd.parent.mkdir(parents=True,exist_ok=True); '
                    'shutil.copytree(s,d,copy_function=os.link) if not d.exists() else None; '
                    'shutil.copyfile(s.parent/"manifest.json",d/"_prepared_from_manifest.json")', remote_run / 'data', cache])
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
                'p.resolve().relative_to(root.resolve()); shutil.rmtree(p)', remote_run, remote_repo/'benchmark/competitors/runs'/series])
            if tool == 'oligoformer' and row['training_seed'] == '2':
                remote(args, ['python3', '-c', 'import shutil,sys; shutil.rmtree(sys.argv[1])', cache])
            completed.add(key)
            print('VERIFIED_AND_RETURNED', tool, run_name, 'seconds', round(time.time()-started, 1), flush=True)


if __name__ == '__main__':
    main()
