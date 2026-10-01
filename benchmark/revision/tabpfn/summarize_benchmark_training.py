"""Summarize audited training and validation only, without held-out scores."""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_validation import checked_bytes, sha256, write_json


def validate_summary(frame):
    expected = set(itertools.product(['grouped', 'random'], range(5), range(3)))
    key = ['axis', 'fold', 'seed']
    if frame.duplicated(key).any() or set(map(tuple, frame[key].to_numpy())) != expected:
        raise ValueError('Expected all 30 unique training identities')
    columns = ['initial_validation_r2', 'selected_validation_r2', 'validation_r2_gain',
               'epochs_completed', 'selected_epoch']
    if not np.isfinite(frame[columns].to_numpy()).all():
        raise ValueError('Nonfinite training summary')
    if not np.allclose(frame.selected_validation_r2-frame.initial_validation_r2, frame.validation_r2_gain, rtol=0, atol=1e-12):
        raise ValueError('Inconsistent validation improvement')
    if not frame.epochs_completed.between(1, 100).all() or not ((frame.selected_epoch >= 0) & (frame.selected_epoch <= frame.epochs_completed)).all():
        raise ValueError('Invalid completed/selected epochs')


def summarize(frame, output):
    validate_summary(frame)
    summary = frame.groupby('axis').agg(
        runs=('seed', 'size'), initial_r2_mean=('initial_validation_r2', 'mean'),
        selected_r2_mean=('selected_validation_r2', 'mean'),
        validation_gain_mean=('validation_r2_gain', 'mean'),
        epochs_min=('epochs_completed', 'min'), epochs_median=('epochs_completed', 'median'),
        epochs_max=('epochs_completed', 'max'),
        initial_checkpoint_selected=('selected_epoch', lambda values: int(values.eq(0).sum())))
    summary.to_csv(output/'training_summary_by_axis.csv')
    frame = frame.sort_values(['axis', 'fold', 'seed']).reset_index(drop=True)
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True)
    positions = np.arange(len(frame))
    axes[0].vlines(positions, frame.initial_validation_r2, frame.selected_validation_r2,
                   color='#b6bdc5', linewidth=1)
    axes[0].scatter(positions, frame.initial_validation_r2, color='#656565', s=24, label='Frozen foundation')
    axes[0].scatter(positions, frame.selected_validation_r2, color='#c56b22', s=24, label='Selected fine-tuning checkpoint')
    axes[0].set_ylabel('Validation R²')
    axes[0].set_title('A  Validation checkpoint selection', loc='left')
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].scatter(positions, frame.epochs_completed, color='#146c94', s=24, label='Completed epochs')
    axes[1].scatter(positions, frame.selected_epoch, color='#c56b22', s=24, label='Selected epoch (0 = initial weights)')
    axes[1].set_ylabel('Epochs')
    axes[1].set_title('B  Training duration and retained checkpoint', loc='left')
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    labels = [f'{"G" if row.axis == "grouped" else "R"}{row.fold+1}/s{row.seed}' for row in frame.itertuples()]
    axes[1].set_xticks(positions, labels, rotation=65, ha='right', fontsize=7)
    axes[1].set_xlabel('G: grouped fold; R: random partition; s: training seed')
    for axis in axes:
        axis.axvline(14.5, color='#999999', linewidth=.7, linestyle='--')
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.2)
    figure.tight_layout()
    for extension in ['png', 'pdf', 'svg']:
        figure.savefig(output/f'training_validation_summary.{extension}', dpi=220, facecolor='white')
    plt.close(figure)
    caption = ('Training/validation diagnostics for 30 TabPFN-3.5 runs. Initial and selected '
               'R² use each partition’s validation records; checkpoints are selected on these '
               'same labels, so improvement is development evidence, not held-out performance. '
               'Validation sets overlap across partitions and are not pooled or treated as '
               'independent experiments. Epoch 0 denotes retained foundation weights. '
               'Grouped fold labels G1–G5 map to stored fold IDs 0–4. No test or HeLa scores are used.')
    (output/'caption.txt').write_text(caption+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new training-summary output directory')
    manifest_path = args.audit/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'verified' or manifest['runs'] != 60:
        raise ValueError('A complete verified audit is required')
    path = args.audit/'training_summary.csv'
    from io import BytesIO
    frame = pd.read_csv(BytesIO(checked_bytes(path, manifest['training_summary_sha256'])))
    args.output.mkdir(parents=True)
    summarize(frame, args.output)
    write_json(args.output/'manifest.json', {'inputs': {str(path): sha256(path), str(manifest_path): sha256(manifest_path)},
               'code_sha256': sha256(Path(__file__)),
               'outputs': {path.name: sha256(path) for path in args.output.iterdir() if path.is_file()}})


if __name__ == '__main__':
    main()
