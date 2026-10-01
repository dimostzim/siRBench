"""Check parallel/resumed cached preprocessing against the frozen serial pipeline."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

revision = Path(__file__).resolve().parents[1]
competitors = revision.parent / "competitors"
source = competitors / "tools/sirnadiscovery/scripts/preprocess.py"
helper = revision / "discovery_structure_cache.py"
spec = importlib.util.spec_from_file_location("cache_helper", helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    frame = pd.DataFrame({"siRNA": ["s1", "s2"], "mRNA": ["m1", "m1"],
                          "siRNA_seq": ["AUGCAUGCAUGCAUGCAUG", "UGCAUGCAUGCAUGCAUGC"],
                          "mRNA_seq": ["AUGC" * 14 + "A"] * 2})
    input_path = directory / "input.csv"
    frame.to_csv(input_path, index=False)
    serial, parallel, cache = [directory / name for name in ("serial", "parallel", "cache")]
    extent = sys.argv[1] if len(sys.argv) > 1 else "57"
    common = ["--input-csv", str(input_path), "--mrna-len", extent]
    subprocess.run([sys.executable, str(source), *common, "--output-dir", str(serial)], check=True)
    command = [sys.executable, str(helper), *common, "--output-dir", str(parallel),
               "--cache-dir", str(cache), "--preprocessor", str(source), "--workers", "2"]
    subprocess.run(command, check=True)
    for filename in ("con_matrix.txt", "self_siRNA_matrix.txt", "self_mRNA_matrix.txt"):
        expected = pd.read_csv(serial / filename, header=None, index_col=0)
        actual = pd.read_csv(parallel / filename, header=None, index_col=0)
        assert expected.index.equals(actual.index), filename
        assert np.array_equal(expected.to_numpy(), actual.to_numpy()), filename
    completion_files = list(cache.glob("*/complete.json"))
    assert len(completion_files) == 5
    before = {str(path): path.stat().st_mtime_ns for path in completion_files}
    subprocess.run(command, check=True)
    after = {str(path): path.stat().st_mtime_ns for path in completion_files}
    assert before == after, "Resume must reuse complete cache entries"
    entry = completion_files[0].parent
    task = json.loads((entry / "complete.json").read_text())["task"]
    with (entry / "vector.npy").open("ab") as handle:
        handle.write(b"invalid")
    try:
        module.validate_cached(entry, task)
    except ValueError as error:
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("Corrupt cache was accepted")
    print(f"PASS: extent {extent}; serial equality, unique sequence cache, resume, corrupt-cache rejection")
