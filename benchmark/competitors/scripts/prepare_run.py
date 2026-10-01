"""Guard an isolated benchmark run against stale inputs, settings, code, or images."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_run(run_dir, inputs, settings):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest = {"settings": settings, "inputs": {
        name: {"source_path": str(path.resolve()), "sha256": sha256(path)}
        for name, path in inputs.items()}}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Run manifest differs; choose a new --run-dir for changed inputs/settings/code/images")
        for name, info in manifest["inputs"].items():
            copied = run_dir / "inputs" / (name + ".csv")
            if not copied.is_file() or sha256(copied) != info["sha256"]:
                raise ValueError("Run input copy was modified or is missing; use a new --run-dir")
        return
    if any(path.name != ".run.lock" for path in run_dir.iterdir()):
        raise ValueError("Refusing to adopt a nonempty run directory without a manifest")
    (run_dir / "inputs").mkdir()
    for name, path in inputs.items():
        shutil.copyfile(path, run_dir / "inputs" / (name + ".csv"))
    # Publish only after all copies are complete. Incomplete initialization is rejected.
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--tool", action="append", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--original", type=int, choices=[0, 1], required=True)
    parser.add_argument("--deterministic", type=int, choices=[0, 1], required=True)
    for name in ["train", "val", "test", "leftout"]:
        parser.add_argument("--" + name, type=Path, required=name != "leftout")
    args = parser.parse_args()
    args.run_dir.resolve().relative_to(args.repo_root.resolve())
    comp = args.repo_root / "benchmark/competitors"
    code = [comp / "run_tool.sh", *sorted((comp / "scripts").glob("*.py"))]
    for tool in args.tool:
        code.extend(path for path in sorted((comp / "tools" / tool).rglob("*"))
                    if path.is_file() and (path.suffix in {".py", ".sh", ".json", ".yaml", ".yml"}
                                           or path.name == "Dockerfile") and ".git" not in path.parts)
    settings = {"seed": args.seed, "original": bool(args.original),
                "deterministic": bool(args.deterministic), "tools": sorted(args.tool),
                "code_sha256": {str(path.relative_to(args.repo_root)): sha256(path) for path in code},
                "docker_images": {tool: subprocess.check_output(
                    ["docker", "image", "inspect", "--format", "{{.Id}}",
                     tool + ":" + os.environ.get("SIRBENCH_IMAGE_TAG", "latest")], text=True).strip()
                    for tool in args.tool}}
    inputs = {name: getattr(args, name) for name in ["train", "val", "test", "leftout"] if getattr(args, name)}
    prepare_run(args.run_dir, inputs, settings)


if __name__ == "__main__":
    main()
