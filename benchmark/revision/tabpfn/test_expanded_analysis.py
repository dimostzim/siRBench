"""Regression checks for preservation of the existing published comparison."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

import verify_expanded_analysis as verifier


@pytest.fixture
def analysis_files(tmp_path):
    original, expanded = tmp_path/'original', tmp_path/'expanded'
    original.mkdir()
    expanded.mkdir()
    manifest = {'methods': ['reference'], 'software_versions': {'numpy': 'fixed'},
                'bootstrap_seed': 20260921, 'inputs': {'records.csv': 'unchanged'}}
    (original/'analysis_manifest.json').write_text(json.dumps(manifest))
    (expanded/'analysis_manifest.json').write_text(json.dumps({**manifest, 'methods': ['reference', *sorted(verifier.NEW_TOOLS)]}))
    labels = np.linspace(0, 1, 3051)
    predictions = labels*0.8 + 0.05
    r2 = float(1-np.square(labels-predictions).sum()/np.square(labels-labels.mean()).sum())
    for filename in verifier.TABLES:
        if filename == 'primary_paired_differences.csv':
            frame = pd.DataFrame([{'left': 'reference', 'right': 'reference', 'estimate_left_minus_right': 0.}])
            added = pd.DataFrame([{'left': 'reference', 'right': 'tabpfn35_frozen', 'estimate_left_minus_right': -0.1}])
        else:
            frame = pd.DataFrame([{'tool': 'reference', 'cohort': 'test', 'metric': 'r2', 'estimate': 0.5}])
            added = pd.DataFrame([{'tool': tool, 'cohort': 'test', 'metric': 'r2', 'estimate': r2} for tool in sorted(verifier.NEW_TOOLS)])
        frame.to_csv(original/filename, index=False)
        pd.concat([frame, added], ignore_index=True).to_csv(expanded/filename, index=False)
    index = []
    for tool in sorted(verifier.NEW_TOOLS):
        for seed in range(3):
            for fold, positions in enumerate(np.array_split(np.arange(3051), 5)):
                path = tmp_path/f'{tool}_{seed}_{fold}.csv'
                pd.DataFrame({'record_id': positions, 'label': labels[positions], 'pred_label': predictions[positions]}).to_csv(path, index=False)
                index.append({'tool': tool, 'axis': 'grouped', 'training_seed': seed, 'fold': fold, 'test_predictions': str(path)})
    index_path = tmp_path/'index.csv'
    pd.DataFrame(index).to_csv(index_path, index=False)
    return original, expanded, index_path, tmp_path/'verification.json'


def run_cli(monkeypatch, paths):
    original, expanded, index, output = paths
    monkeypatch.setattr(sys, 'argv', ['verify', '--original', str(original), '--expanded', str(expanded),
                                    '--tabpfn-index', str(index), '--output', str(output)])
    verifier.main()


def test_complete_verification(monkeypatch, analysis_files):
    run_cli(monkeypatch, analysis_files)
    result = json.loads(analysis_files[-1].read_text())
    assert result['status'] == 'verified'
    assert set(result['independent_grouped_r2']) == verifier.NEW_TOOLS


def test_changed_original_metric_is_rejected(monkeypatch, analysis_files):
    path = analysis_files[1]/'metric_summary.csv'
    frame = pd.read_csv(path)
    frame.loc[0, 'estimate'] = 0.51
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        run_cli(monkeypatch, analysis_files)
    assert not analysis_files[-1].exists()


def test_changed_original_input_is_rejected(monkeypatch, analysis_files):
    path = analysis_files[1]/'analysis_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['inputs']['records.csv'] = 'changed'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Original input changed'):
        run_cli(monkeypatch, analysis_files)


def test_incorrect_tabpfn_r2_is_rejected(monkeypatch, analysis_files):
    path = analysis_files[1]/'primary_group_bootstrap.csv'
    frame = pd.read_csv(path)
    frame.loc[frame.tool == 'tabpfn35_frozen', 'estimate'] += 0.01
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Independent pooled R2 differs'):
        run_cli(monkeypatch, analysis_files)
