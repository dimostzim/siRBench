import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1]/"run_oligo_bert_matrix.py"
spec = importlib.util.spec_from_file_location("matrix_controller",PATH)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def check(run, tool="oligoformer", original=False, required=True):
    return subprocess.run([sys.executable,"-c",controller.CHECK_TRAINING_COMPLETE,str(run),tool,"2",str(int(original)),str(int(required))],capture_output=True,text=True)


def test_empty_run_allowed_only_before_training(tmp_path):
    assert check(tmp_path,required=False).returncode == 0
    result = check(tmp_path)
    assert result.returncode != 0 and "checkpoint is missing" in result.stderr


def test_checkpoint_without_completion_metadata_is_rejected(tmp_path):
    directory = tmp_path/"models"/"oligoformer"
    directory.mkdir(parents=True)
    (directory/"model.pt").write_bytes(b"checkpoint")
    result = check(tmp_path,required=False)
    assert result.returncode != 0 and "without completed training metadata" in result.stderr


@pytest.mark.parametrize("tool,original,epochs,patience,metric",[
    ("oligoformer",False,100,20,"r2"),
    ("sirnabert",False,100,20,"val_r2"),
    ("oligoformer",True,200,30,"loss+auc"),
    ("sirnabert",True,30,0,"val_loss"),
])
def test_completed_training_settings_are_checked(tmp_path,tool,original,epochs,patience,metric):
    directory = tmp_path/"models"/tool
    directory.mkdir(parents=True)
    (directory/"model.pt").write_bytes(b"checkpoint")
    settings = dict(seed=2,epochs=epochs,early_stopping=patience,early_stop_metric=metric,best_epoch=5)
    if tool=="sirnabert" and original:
        settings.update(checkpoint_selection="final_epoch",best_epoch=29,epochs_completed=30)
    metadata = directory/"train_meta.json"
    metadata.write_text(json.dumps(settings))
    assert check(tmp_path,tool,original).returncode == 0
    settings["seed"] = 1
    metadata.write_text(json.dumps(settings))
    result = check(tmp_path,tool,original)
    assert result.returncode != 0 and "differs from requested settings" in result.stderr


def test_original_bert_rejects_best_validation_instead_of_final_epoch(tmp_path):
    directory = tmp_path/"models"/"sirnabert"
    directory.mkdir(parents=True)
    (directory/"model.pt").write_bytes(b"checkpoint")
    settings = dict(seed=2,epochs=30,early_stopping=0,early_stop_metric="val_loss",best_epoch=5,
                    checkpoint_selection="best_validation",epochs_completed=30)
    (directory/"train_meta.json").write_text(json.dumps(settings))
    result = check(tmp_path,"sirnabert",True)
    assert result.returncode != 0 and "final epoch checkpoint" in result.stderr
