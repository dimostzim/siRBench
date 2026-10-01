import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "competitors/tools/ensirna/test.py"
spec = importlib.util.spec_from_file_location("ensirna_test", MODULE)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


@pytest.mark.parametrize("scores", [(0.9, 0.3), (0.1, 0.7)])
def test_selects_trainer_ranked_best_for_both_metric_directions(tmp_path, scores):
    paths = [str(tmp_path / "best.ckpt"), str(tmp_path / "worse.ckpt")]
    (tmp_path / "topk_map.txt").write_text(f"{scores[0]}: {paths[0]}\n{scores[1]}: {paths[1]}\n")
    assert wrapper.select_checkpoint(paths) == paths[0]


def test_refuses_implicit_ensemble_across_training_runs(tmp_path):
    with pytest.raises(ValueError, match="one training run"):
        wrapper.select_checkpoint([str(tmp_path / "run1/a.ckpt"), str(tmp_path / "run2/b.ckpt")])
