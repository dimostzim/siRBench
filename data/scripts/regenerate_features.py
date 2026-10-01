"""Regenerate sequence features in isolated work directories, preserving base metadata."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import make_all_features


def build_chunk(frame):
    original_directory = Path.cwd()
    # RNAup writes auxiliary files in cwd; each process needs a private directory.
    with tempfile.TemporaryDirectory(prefix="sirbench-features-") as work:
        try:
            os.chdir(work)
            return make_all_features.build_unified_features(frame)
        finally:
            os.chdir(original_directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    frame = pd.read_csv(args.input, dtype=str)
    versions = {tool: subprocess.check_output([tool, "--version"], text=True).strip()
                for tool in ["RNAfold", "RNAcofold", "RNAup"]}
    chunks = [frame.iloc[index].copy() for index in np.array_split(np.arange(len(frame)), args.workers)
              if len(index)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        output = pd.concat(pool.map(build_chunk, chunks), ignore_index=True)
    if not output.columns.is_unique or not output[frame.columns].equals(frame):
        raise ValueError("Feature regeneration changed metadata or duplicated columns")
    feature_columns = output.columns.difference(frame.columns)
    if len(feature_columns) != 100 or not np.isfinite(output[feature_columns].to_numpy()).all():
        raise ValueError("Expected 100 finite feature columns")
    output[feature_columns] = output[feature_columns].round(3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.csv")
    output.to_csv(temporary, index=False)
    temporary.replace(args.output)
    manifest = {"rows": len(output), "features": list(feature_columns), "versions": versions,
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [
                    args.input, Path(__file__), SCRIPTS / "make_all_features.py",
                    SCRIPTS / "features_calculator.py"]}}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(output)} rows and {len(feature_columns)} features: {args.output}")


if __name__ == "__main__":
    main()
