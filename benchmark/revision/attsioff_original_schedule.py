#!/usr/bin/env python3
"""Isolated released AttSiOff stopping semantics; frozen primary trainer is unchanged."""
import hashlib
from pathlib import Path
import sys

FROZEN_TRAINER_SHA256 = "939f8fdc8b7c69a0c6a88505a6ce534cc6703df94ad2b0f07a46c6cff9208377"


def released_score(value):
    return round(float("nan") if value is None else value, 3)


def released_improvement(score, best):
    return best is None or not (score < best)


def effective_source(source):
    previous = """        else:
            cur_metric = val_spcc
            improved = best_metric is None or (cur_metric is not None and cur_metric > best_metric)
"""
    replacement = """        else:
            cur_metric = released_score(val_spcc)
            improved = released_improvement(cur_metric, best_metric)
"""
    if source.count(previous) != 1:
        raise ValueError("Frozen Spearman selector does not match expected source")
    source = source.replace(previous, replacement)
    previous_meta = '        "early_stop_metric": args.early_stop_metric,'
    replacement_meta = previous_meta + '\n        "released_selection": {"round_decimals": 3, "ties_replace_checkpoint_and_reset_patience": True},'
    if source.count(previous_meta) != 1:
        raise ValueError("Frozen metadata does not match expected source")
    return source.replace(previous_meta, replacement_meta)


def main():
    if "--original-params" not in sys.argv:
        raise ValueError("This sensitivity requires the released1000-epoch/patience20 configuration")
    frozen = Path(__file__).resolve().parents[1] / "competitors/tools/attsioff/train.py"
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != FROZEN_TRAINER_SHA256:
        raise ValueError("Frozen primary trainer checksum changed")
    source = effective_source(frozen.read_text())
    model_dir = Path(sys.argv[sys.argv.index("--model-dir") + 1])
    archive = model_dir.parents[1] / "runtime_sources"
    archive.mkdir(parents=True, exist_ok=True)
    effective = archive / "train_original_schedule_effective.py"
    effective.write_text(source)
    namespace = {"__name__": "__main__", "__file__": str(frozen),
                 "released_score": released_score, "released_improvement": released_improvement}
    exec(compile(source, str(effective), "exec"), namespace)


if __name__ == "__main__":
    main()
