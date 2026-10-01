import importlib.util
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "competitors/tools/ensirna/train.py"
spec = importlib.util.spec_from_file_location("ensirna_train", MODULE)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def run_wrapper(monkeypatch, tmp_path, extra):
    calls = []
    monkeypatch.setattr(wrapper.sys, "argv", [str(MODULE), "--train-set", "train.jsonl",
                         "--valid-set", "val.jsonl", "--model-dir", str(tmp_path), *extra])
    monkeypatch.setattr(wrapper.subprocess, "check_call", lambda command, **kwargs: calls.append(command))
    monkeypatch.setenv("ENSIRNA_SEED", "12")
    wrapper.main()
    command = calls[0]
    assert "--shuffle" in command
    return {flag: command[index + 1] for index, flag in enumerate(command[:-1]) if flag.startswith("--")}


def test_original_configuration_matches_released_config(monkeypatch, tmp_path):
    args = run_wrapper(monkeypatch, tmp_path, ["--original-params"])
    expected = {"--lr": "0.0001", "--final_lr": "1e-05", "--max_epoch": "100",
                "--batch_size": "16", "--embed_dim": "128", "--save_topk": "10",
                "--hidden_size": "256", "--n_layers": "2", "--k_neighbors": "9", "--num_workers": "4"}
    assert {name: args[name] for name in expected} == expected


def test_benchmark_overrides_are_forwarded(monkeypatch, tmp_path):
    args = run_wrapper(monkeypatch, tmp_path, ["--lr", "0.0002", "--final-lr", "0.00002",
                       "--max-epoch", "40", "--seed", "73"])
    assert (args["--lr"], args["--final_lr"], args["--max_epoch"], wrapper.os.environ["ENSIRNA_SEED"]) == (
        "0.0002", "2e-05", "40", "73")


def test_standalone_defaults_match_released_learning_schedule(monkeypatch, tmp_path):
    args = run_wrapper(monkeypatch, tmp_path, [])
    assert (args["--lr"], args["--final_lr"], args["--max_epoch"]) == ("0.0001", "1e-05", "100")
