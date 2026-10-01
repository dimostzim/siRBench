import importlib.util
import json
from pathlib import Path
import sys
import threading

import pytest

TOOL = Path(__file__).resolve().parents[2] / 'competitors/tools/ensirna'
sys.path.insert(0, str(TOOL))
spec = importlib.util.spec_from_file_location('ensirna_matrix', TOOL / 'run_matrix.py')
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


def test_drain_marker_preserves_two_active_jobs(tmp_path):
    marker = tmp_path / 'drain'
    barrier = threading.Barrier(2)
    called = []

    def run(item):
        called.append(item)
        barrier.wait(timeout=10)
        marker.touch()

    matrix.run_bounded(range(6), run, jobs=2, stop_file=marker)
    assert sorted(called) == [0, 1]


def test_preexisting_marker_schedules_nothing(tmp_path):
    marker = tmp_path / 'drain'
    marker.touch()
    called = []
    matrix.run_bounded(range(6), called.append, jobs=2, stop_file=marker)
    assert called == []


def test_nonfinite_prediction_is_rejected(tmp_path):
    expected = tmp_path / 'test.csv'
    expected.write_text('record_id,efficiency\na,0.2\n')
    prediction = tmp_path / 'preds.csv'
    prediction.write_text('id,label,pred_label\na,0.2,nan\n')
    with pytest.raises(ValueError, match='Nonfinite'):
        matrix.validate_predictions(prediction, expected)


def test_training_metadata_requires_ranked_checkpoint_and_frozen_architecture(tmp_path):
    checkpoint = tmp_path / 'models/checkpoint/epoch3.ckpt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    topk = checkpoint.parent / 'topk_map.txt'
    topk.write_text('0.3: /work/models/checkpoint/epoch3.ckpt\n')
    metadata = {'config': {'seed': 1, 'lr': 1e-4, 'final_lr': 1e-5, 'max_epoch': 100,
        'batch_size': 16, 'embed_dim': 128, 'hidden_size': 256, 'n_layers': 2,
        'k_neighbors': 9, 'shuffle': True, 'num_workers': 4, 'prefetch_factor': 1,
        'val_metric': 'r2', 'patience': 20, 'legacy_stopping': False,
        'metric_min_better': False, 'step_per_epoch': 10},
        'completed_epochs': 24, 'completed_steps': 240, 'best_validation_metric': 0.3,
        'selected_checkpoint': '/work/models/checkpoint/epoch3.ckpt'}
    path = tmp_path / 'training_metadata.json'
    path.write_text(json.dumps(metadata))
    row = {'training_seed': 1, 'train_meta': str(path)}
    matrix.validate_training_metadata(row, tmp_path, False)
    topk.write_text('0.4: /work/models/checkpoint/epoch4.ckpt\n')
    with pytest.raises(ValueError, match='trainer-ranked'):
        matrix.validate_training_metadata(row, tmp_path, False)
    metadata['config']['hidden_size'] = 128
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='frozen policy'):
        matrix.validate_training_metadata(row, tmp_path, False)
