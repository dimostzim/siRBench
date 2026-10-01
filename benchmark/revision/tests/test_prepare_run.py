import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "competitors/scripts"))
from prepare_run import prepare_run


def inputs(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("siRNA,efficiency\nAAAA,0.5\n")
    return {"train": source, "val": source, "test": source}


def test_identical_run_resumes_with_verified_copies(tmp_path):
    sources = inputs(tmp_path)
    run = tmp_path / "run"
    prepare_run(run, sources, {"seed": 42, "original": False})
    prepare_run(run, sources, {"seed": 42, "original": False})
    assert (run / "inputs/train.csv").read_bytes() == sources["train"].read_bytes()


@pytest.mark.parametrize("changed", [{"seed": 43, "original": False}, {"seed": 42, "original": True}])
def test_changed_configuration_cannot_reuse_run(tmp_path, changed):
    sources = inputs(tmp_path)
    run = tmp_path / "run"
    prepare_run(run, sources, {"seed": 42, "original": False})
    with pytest.raises(ValueError, match="manifest differs"):
        prepare_run(run, sources, changed)


def test_changed_input_cannot_reuse_run(tmp_path):
    sources = inputs(tmp_path)
    run = tmp_path / "run"
    prepare_run(run, sources, {})
    sources["train"].write_text("different data")
    with pytest.raises(ValueError, match="manifest differs"):
        prepare_run(run, sources, {})


def test_modified_input_copy_is_detected(tmp_path):
    sources = inputs(tmp_path)
    run = tmp_path / "run"
    prepare_run(run, sources, {})
    (run / "inputs/train.csv").write_text("different data")
    with pytest.raises(ValueError, match="input copy"):
        prepare_run(run, sources, {})


def test_unmanifested_cache_is_not_adopted(tmp_path):
    sources = inputs(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "model.pt").write_text("old model")
    with pytest.raises(ValueError, match="nonempty"):
        prepare_run(run, sources, {})


def test_shell_entrypoint_isolates_original_mode_and_rejects_changed_seed(tmp_path):
    import json
    import os
    import shutil
    import subprocess

    original = Path(__file__).resolve().parents[2] / "competitors"
    repo = tmp_path / "repo"
    competitors = repo / "benchmark/competitors"
    (competitors / "scripts").mkdir(parents=True)
    shutil.copyfile(original / "run_tool.sh", competitors / "run_tool.sh")
    shutil.copyfile(original / "scripts/prepare_run.py", competitors / "scripts/prepare_run.py")
    commands = tmp_path / "commands.jsonl"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(f'''#!{sys.executable}
import json, os, sys
if sys.argv[1] == "-c" or sys.argv[1].endswith("prepare_run.py"):
    os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])
with open(os.environ["RUN_TEST_LOG"], "a") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")
''')
    fake_python.chmod(0o755)
    fake_docker = fake_bin / "docker"
    fake_docker.write_text("#!/bin/sh\nprintf 'sha256:test-image\\n'\n")
    fake_docker.chmod(0o755)
    env = {**os.environ, "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"], "RUN_TEST_LOG": str(commands)}
    source = inputs(tmp_path)["train"]
    command = ["bash", str(competitors / "run_tool.sh"), "--tool", "sirnabert",
               "--train", str(source), "--val", str(source), "--test", str(source)]
    first, second = repo / "runs/standard", repo / "runs/original"
    subprocess.run(command + ["--run-dir", str(first)], env=env, check=True, capture_output=True)
    subprocess.run(command + ["--run-dir", str(second), "--original"], env=env, check=True, capture_output=True)
    calls = [json.loads(line) for line in commands.read_text().splitlines()]
    training = [call for call in calls if call[0] == "scripts/train.py"]
    assert len(training) == 2
    assert str(first / "models/sirnabert") in training[0]
    assert str(second / "models/sirnabert") in training[1]
    assert "--original-params" not in training[0] and "--original-params" in training[1]
    failed = subprocess.run(command + ["--run-dir", str(first), "--seed", "43"], env=env, capture_output=True, text=True)
    assert failed.returncode != 0 and "manifest differs" in failed.stderr
    assert commands.read_text().count("scripts/train.py") == 2
