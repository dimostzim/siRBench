"""Compare rounded/tie-aware selection directly with the released stopping class."""
import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attsioff_original_schedule import released_score, released_improvement


def upstream_stopping(saved):
    source = Path(__file__).resolve().parents[2] / 'competitors/tools/attsioff/attsioff_src/utils.py'
    tree = ast.parse(source.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EarlyStopping')
    namespace = {'os': os, 'torch': SimpleNamespace(save=lambda state, path: saved.append(state))}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['EarlyStopping']('/unused', 'model.pt', patience=20, greater=True)


@pytest.mark.parametrize('scores', [
    [0.50041, 0.50049, 0.50042] + [0.49] * 20,
    [0.5] * 1000,
    [None, None, 0.2] + [0.1] * 20,
])
def test_selector_matches_released_checkpoint_and_stopping(scores):
    saved = []
    upstream = upstream_stopping(saved)
    best, best_epoch = None, -1
    for epoch, raw in enumerate(scores):
        score = released_score(raw)
        upstream(score, epoch, SimpleNamespace(state_dict=lambda: epoch))
        if released_improvement(score, best):
            best, best_epoch = score, epoch
        assert saved[-1] == best_epoch
        assert upstream.early_stop == (epoch - best_epoch >= 20)
        if upstream.early_stop:
            break
    if len(scores) == 1000:
        assert best_epoch == 999
        assert not upstream.early_stop
