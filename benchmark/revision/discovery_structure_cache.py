#!/usr/bin/env python3
"""Resumable bounded-worker structure preprocessing using the frozen wrapper math."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_preprocessor(path):
    spec = importlib.util.spec_from_file_location("frozen_discovery_preprocess", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_key(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()


def validate_cached(directory, task):
    meta_path = directory / "complete.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    if meta["task"] != task:
        raise ValueError(f"Cache task mismatch: {directory}")
    for name, digest in meta["sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Cache checksum mismatch: {directory / name}")
    values = np.load(directory / "vector.npy", allow_pickle=False)
    if values.shape != (task["components"],) or not np.isfinite(values).all():
        raise ValueError(f"Invalid cached vector: {directory}")
    return True


def compute_task(arguments):
    task, source_path, cache_root = arguments
    key = task_key(task)
    directory = Path(cache_root) / key
    if validate_cached(directory, task):
        return key, "cached"
    module = load_preprocessor(source_path)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="work-", dir=directory) as work:
        if task["kind"] == "interaction":
            dotplot = module._run_rnacofold(task["sequence"], key, work)
        else:
            dotplot = module._run_rnafold(task["sequence"], key, work)
        matrix = module._parse_dp_ps(dotplot, task["size"])
        if task["kind"] == "interaction":
            # Preserve the upstream two-position deletion, including its known indexing issue.
            mask = np.ones(task["size"], dtype=bool)
            start = task["sirna_len"]
            mask[start:start + 2] = False
            matrix = matrix[mask][:, mask]
        vector = module._reduce_matrix(matrix, task["components"])
        if vector.shape != (task["components"],) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid generated vector: {key}")
        del matrix
        np.save(Path(work) / "vector.npy", vector, allow_pickle=False)
        with open(dotplot, "rb") as source, gzip.open(Path(work) / "dotplot.ps.gz", "wb") as target:
            shutil.copyfileobj(source, target)
        for name in ("vector.npy", "dotplot.ps.gz"):
            os.replace(Path(work) / name, directory / name)
    meta = {"task": task, "elapsed_seconds": time.monotonic() - started,
            "sha256": {name: sha256(directory / name) for name in ("vector.npy", "dotplot.ps.gz")}}
    temporary_meta = directory / "complete.json.tmp"
    temporary_meta.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(temporary_meta, directory / "complete.json")
    return key, "computed"


def build_tasks(frame, params, source_sha, versions):
    tasks, mapping = {}, {"self_siRNA_matrix.txt": {}, "self_mRNA_matrix.txt": {}, "con_matrix.txt": {}}

    def add(filename, identifier, kind, sequence, size, components):
        task = {"kind": kind, "sequence": sequence, "size": size, "components": components,
                "sirna_len": params.sirna_len, "source_sha256": source_sha, "versions": versions}
        key = task_key(task)
        old = mapping[filename].setdefault(str(identifier), key)
        if old != key:
            raise ValueError(f"Identifier maps to conflicting sequences: {identifier}")
        tasks[key] = task

    for row in frame.itertuples(index=False):
        sirna, mrna = str(row.siRNA_seq).upper().replace("T", "U"), str(row.mRNA_seq).upper().replace("T", "U")
        if len(sirna) > params.sirna_len or len(mrna) > params.mrna_len:
            raise ValueError(f"Sequence exceeds requested representation: {row.siRNA}/{row.mRNA}")
        add("self_siRNA_matrix.txt", row.siRNA, "sirna", sirna, params.sirna_len, params.sirna_svd)
        add("self_mRNA_matrix.txt", row.mRNA, "mrna", mrna, params.mrna_len, params.mrna_svd)
        add("con_matrix.txt", f"{row.siRNA}_{row.mRNA}", "interaction", f"{sirna}&{mrna}",
            params.sirna_len + params.mrna_len + 2, params.con_svd)
    return tasks, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="Prepared Discovery CSV with stable IDs and RNA sequences.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--preprocessor", required=True, help="Frozen wrapper scripts/preprocess.py.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sirna-len", type=int, default=19)
    parser.add_argument("--mrna-len", type=int, default=9756)
    parser.add_argument("--sirna-svd", type=int, default=6)
    parser.add_argument("--mrna-svd", type=int, default=100)
    parser.add_argument("--con-svd", type=int, default=50)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    versions = {name: subprocess.check_output([name, "--version"], text=True).strip()
                for name in ("RNAfold", "RNAcofold")}
    versions.update(numpy=np.__version__)
    import sklearn
    import scipy
    versions.update(sklearn=sklearn.__version__, scipy=scipy.__version__)
    frame = pd.read_csv(args.input_csv)
    tasks, mapping = build_tasks(frame, args, sha256(args.preprocessor), versions)
    manifest = {"input_sha256": sha256(args.input_csv), "source_sha256": sha256(args.preprocessor),
                "helper_sha256": sha256(__file__), "versions": versions, "rows": len(frame),
                "task_count": len(tasks), "mapping": mapping}
    manifest_path = output / "input_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing output has a different input or implementation manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    work = [(task, str(Path(args.preprocessor).resolve()), str(Path(args.cache_dir).resolve())) for task in tasks.values()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(compute_task, arguments) for arguments in work]
        for number, future in enumerate(as_completed(futures), 1):
            key, status = future.result()
            print(f"{number}/{len(tasks)} {status} {key}", flush=True)
    for filename, entries in mapping.items():
        values = {identifier: np.load(Path(args.cache_dir) / key / "vector.npy", allow_pickle=False)
                  for identifier, key in entries.items()}
        temporary_output = output / (filename + ".tmp")
        pd.DataFrame.from_dict(values, orient="index").to_csv(temporary_output, header=False)
        os.replace(temporary_output, output / filename)
    complete = {"input_manifest_sha256": sha256(manifest_path),
                "sha256": {name: sha256(output / name) for name in mapping}}
    (output / "complete.json").write_text(json.dumps(complete, indent=2) + "\n")


if __name__ == "__main__":
    main()
