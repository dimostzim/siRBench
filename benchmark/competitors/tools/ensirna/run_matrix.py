#!/usr/bin/env python3
"""Run the frozen ENsiRNA matrix with canonical per-record feature reuse."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from subset_features import sha256


def validate_predictions(predictions, expected_csv):
    with predictions.open() as handle:
        actual = list(csv.DictReader(handle))
    with expected_csv.open() as handle:
        expected = list(csv.DictReader(handle))
    if [row['id'] for row in actual] != [row['record_id'] for row in expected]:
        raise ValueError(f'Prediction identity/order mismatch: {predictions}')
    if [float(row['label']) for row in actual] != [float(row['efficiency']) for row in expected]:
        raise ValueError(f'Prediction label mismatch: {predictions}')
    if not all(math.isfinite(float(row['pred_label'])) for row in actual):
        raise ValueError(f'Nonfinite predictions: {predictions}')


def validate_training_metadata(row, repo, original):
    metadata = json.loads(Path(row['train_meta']).read_text())
    config = metadata['config']
    expected = {'seed': int(row['training_seed']), 'lr': 1e-4, 'final_lr': 1e-5,
                'max_epoch': 100, 'batch_size': 16, 'embed_dim': 128,
                'hidden_size': 256, 'n_layers': 2, 'k_neighbors': 9, 'shuffle': True,
                'num_workers': 4, 'prefetch_factor': 1,
                'val_metric': 'loss' if original else 'r2',
                'patience': 1000 if original else 20, 'legacy_stopping': original,
                'metric_min_better': original}
    mismatches = {name: (config.get(name), value) for name, value in expected.items()
                  if config.get(name) != value}
    if mismatches:
        raise ValueError(f'Training settings differ from the frozen policy: {mismatches}')
    epochs = metadata['completed_epochs']
    if not 1 <= epochs <= 100 or (original and epochs != 100):
        raise ValueError('Unexpected completed epoch count')
    if metadata['completed_steps'] != epochs * config['step_per_epoch']:
        raise ValueError('Completed training step count is inconsistent')
    checkpoint = repo / Path(metadata['selected_checkpoint']).relative_to('/work')
    if not checkpoint.is_file() or not math.isfinite(metadata['best_validation_metric']):
        raise ValueError('Selected checkpoint or validation score is invalid')
    score, selected = (checkpoint.parent / 'topk_map.txt').read_text().splitlines()[0].split(':', 1)
    if Path(selected.strip()).name != checkpoint.name or float(score) != metadata['best_validation_metric']:
        raise ValueError('Final metadata does not identify the trainer-ranked best checkpoint')


def run_bounded(rows, run_one, jobs, stop_file=None):
    pending = iter(rows)
    active = set()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        def fill_slots():
            while len(active) < jobs:
                if stop_file and stop_file.exists():
                    return
                row = next(pending, None)
                if row is None:
                    return
                active.add(pool.submit(run_one, row))

        fill_slots()
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                active.remove(future)
                future.result()
            fill_slots()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--canonical-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--axes', nargs='+', choices=['grouped', 'random', 'standardized', 'original'], default=['grouped', 'random'])
    parser.add_argument('--folds', nargs='+', type=int, choices=range(5), default=list(range(5)))
    parser.add_argument('--hela-csv', type=Path)
    parser.add_argument('--original-params', action='store_true')
    parser.add_argument('--jobs', type=int, choices=[1, 2], default=1)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    tool = Path(__file__).resolve().parent
    competitors = tool.parents[1]

    def container_path(path):
        return str(Path('/work') / path.resolve().relative_to(repo))

    container = ['docker', 'run', '--rm', '--entrypoint', 'python3', '-u', f'{os.getuid()}:{os.getgid()}',
                 '-v', f'{repo}:/work', '-w', container_path(tool), '-e', 'OMP_NUM_THREADS=4',
                 '-e', 'MKL_NUM_THREADS=4', 'ensirna:revision']
    manifest = args.canonical_dir / 'feature_manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    rows = []
    for axis in args.axes:
        for fold in args.folds:
            for seed in range(3):
                run = args.output / f'{axis}_fold_{fold}_seed_{seed}'
                rows.append(dict(tool='ensirna', axis=axis, fold=fold, training_seed=seed, run_dir=str(run),
                    test_predictions=str(run / 'results/ensirna/preds.csv'),
                    hela_predictions=str(run / 'results/ensirna/preds_leftout.csv'),
                    train_meta=str(run / 'models/ensirna/version_0/training_metadata.json'), status='pending'))

    if len({row['run_dir'] for row in rows}) != len(rows):
        raise ValueError('The run plan contains duplicate axes or folds')
    index_lock = threading.Lock()

    def write_index():
        with index_lock:
            temporary = args.index.with_suffix('.tmp')
            with temporary.open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(args.index)

    write_index()
    def run_one(row):
        run = Path(row['run_dir'])
        fold = args.protocol / row['axis'] / f"fold_{row['fold']}"
        inputs = {part: fold / f'{part}.csv' for part in ['train', 'val', 'test']}
        inputs['leftout'] = args.hela_csv or args.protocol / 'hela_full.csv'
        manifest_command = [sys.executable, str(competitors / 'scripts/prepare_run.py'),
            '--repo-root', str(repo), '--run-dir', str(run), '--tool', 'ensirna',
            '--seed', str(row['training_seed']), '--original', str(int(args.original_params)), '--deterministic', '0']
        for part, path in inputs.items():
            manifest_command.extend([f'--{part}', str(path)])
        subprocess.run(manifest_command, check=True)
        prediction_paths = [Path(row['test_predictions']), Path(row['hela_predictions'])]
        completed_outputs = prediction_paths + [run / "results/ensirna/metrics.json", run / "results/ensirna/metrics_leftout.json", Path(row["train_meta"])]
        if all(path.is_file() for path in completed_outputs):
            validate_predictions(prediction_paths[0], inputs['test'])
            validate_predictions(prediction_paths[1], inputs['leftout'])
            validate_training_metadata(row, repo, args.original_params)
            row['status'] = 'complete'
            write_index()
            return
        if any((run / 'models').rglob('*.ckpt')):
            raise RuntimeError(f'Incomplete run contains checkpoints; inspect before resuming: {run}')
        row['status'] = 'preparing'
        write_index()
        try:
            for part in inputs:
                source = run / f'inputs/{part}.csv'
                output = run / f'data/ensirna/{part}.jsonl'
                if output.exists():
                    prior = json.loads((output.with_name(output.stem + '_processed') / 'revision_manifest.json').read_text())
                    if prior['split_csv_sha256'] != sha256(source) or prior['canonical_manifest_sha256'] != sha256(manifest):
                        raise ValueError(f'Prepared feature manifest changed: {output}')
                    continue
                command = container + ['subset_features.py',
                    '--canonical-jsonl', container_path(args.canonical_dir / 'data/all.jsonl'),
                    '--canonical-processed', container_path(args.canonical_dir / 'data/all_processed'),
                    '--manifest', container_path(manifest), '--split-csv', container_path(source),
                    '--output-jsonl', container_path(output)]
                subprocess.run(command, check=True)
            row['status'] = 'running'
            write_index()
            command = ['bash', str(competitors / 'run_tool.sh'), '--tool', 'ensirna', '--run-dir', str(run),
                       '--seed', str(row['training_seed'])]
            if args.original_params:
                command.append('--original')
            for part, path in inputs.items():
                command.extend([f'--{part}', str(path)])
            with (run / 'run.log').open('w') as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True,
                               env={**os.environ, 'QUIET': '0', 'TMPDIR': str(run)})
            validate_predictions(prediction_paths[0], inputs['test'])
            validate_predictions(prediction_paths[1], inputs['leftout'])
            validate_training_metadata(row, repo, args.original_params)
            row['status'] = 'complete'
        except Exception:
            row['status'] = 'failed'
            write_index()
            raise
        write_index()
        print(f"Completed {row['axis']} fold {row['fold']} seed {row['training_seed']}", flush=True)


    run_bounded(rows, run_one, args.jobs, args.stop_file)
    completed = sum(row['status'] == 'complete' for row in rows)
    print(f'Verified {completed}/{len(rows)} planned runs; unscheduled rows remain pending', flush=True)


if __name__ == '__main__':
    main()
