#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def _safe_id(raw):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(raw))
    return safe or "id"


def _ensure_tool(tool):
    if shutil.which(tool) is None:
        raise FileNotFoundError(f"{tool} not found on PATH. Install ViennaRNA to provide {tool}.")


def _parse_dp_ps(path, size):
    mat = np.zeros((size, size), dtype=np.float32)
    started = False
    with open(path, "r") as f:
        for line in f:
            if "start of base pair probability data" in line:
                started = True
                continue
            if not started:
                continue
            if "ubox" in line:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    i = int(parts[0]) - 1
                    j = int(parts[1]) - 1
                    prob = float(parts[2])
                except Exception:
                    continue
                if not np.isfinite(prob):
                    continue
                if 0 <= i < size and 0 <= j < size:
                    mat[i, j] = prob * prob
    mat = mat + mat.T - np.diag(np.diag(mat))
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = np.clip(mat, 0.0, 1.0)
    return mat


def _reduce_matrix(mat, n_components):
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    min_dim = min(mat.shape)
    if min_dim <= 1:
        vec = np.zeros(n_components, dtype=np.float32)
        return vec
    k = max(1, min(n_components, min_dim - 1))
    svd = TruncatedSVD(n_components=k, random_state=0)
    try:
        reduced = svd.fit_transform(csr_matrix(mat))
    except ValueError:
        # Fallback for unexpected NaN/Inf propagation during SVD
        return np.zeros(n_components, dtype=np.float32)
    vec = reduced.mean(axis=0)
    if k < n_components:
        vec = np.pad(vec, (0, n_components - k), mode="constant")
    return vec.astype(np.float32)


def _run_rnafold(seq, out_prefix, work_dir):
    cmd = ["RNAfold", "-p", "--id-prefix", out_prefix]
    subprocess.run(cmd, input=seq + "\n", text=True, cwd=work_dir, check=True, stdout=subprocess.DEVNULL)
    return _resolve_dp_ps(work_dir, out_prefix)


def _run_rnacofold(seq_pair, out_prefix, work_dir):
    cmd = ["RNAcofold", "-p", "--id-prefix", out_prefix]
    subprocess.run(cmd, input=seq_pair + "\n", text=True, cwd=work_dir, check=True, stdout=subprocess.DEVNULL)
    return _resolve_dp_ps(work_dir, out_prefix)


def _resolve_dp_ps(work_dir, out_prefix):
    direct = os.path.join(work_dir, f"{out_prefix}_dp.ps")
    if os.path.exists(direct):
        return direct
    import glob
    matches = sorted(glob.glob(os.path.join(work_dir, f"{out_prefix}_*_dp.ps")))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing dp.ps for prefix {out_prefix} in {work_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True, help="CSV with columns: siRNA, mRNA, siRNA_seq, mRNA_seq")
    p.add_argument("--output-dir", required=True, help="Directory to write siRNA_split_preprocess files")
    p.add_argument("--sirna-len", type=int, default=19)
    p.add_argument("--mrna-len", type=int, default=57)
    p.add_argument("--sirna-svd", type=int, default=6)
    p.add_argument("--mrna-svd", type=int, default=100)
    p.add_argument("--con-svd", type=int, default=50)
    args = p.parse_args()

    _ensure_tool("RNAfold")
    _ensure_tool("RNAcofold")

    df = pd.read_csv(args.input_csv)
    req_cols = {"siRNA", "mRNA", "siRNA_seq", "mRNA_seq"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["siRNA_seq"] = df["siRNA_seq"].astype(str).str.upper().str.replace("T", "U")
    df["mRNA_seq"] = df["mRNA_seq"].astype(str).str.upper().str.replace("T", "U")

    os.makedirs(args.output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        # self-fold features for siRNA
        sirna_vecs = {}
        for sirna_id, seq in df[["siRNA", "siRNA_seq"]].drop_duplicates().itertuples(index=False):
            sid = _safe_id(sirna_id)
            dp = _run_rnafold(seq, sid, tmp)
            mat = _parse_dp_ps(dp, args.sirna_len)
            vec = _reduce_matrix(mat, args.sirna_svd)
            sirna_vecs[sirna_id] = vec

        # self-fold features for mRNA
        mrna_vecs = {}
        for mrna_id, seq in df[["mRNA", "mRNA_seq"]].drop_duplicates().itertuples(index=False):
            mid = _safe_id(mrna_id)
            dp = _run_rnafold(seq, mid, tmp)
            mat = _parse_dp_ps(dp, args.mrna_len)
            vec = _reduce_matrix(mat, args.mrna_svd)
            mrna_vecs[mrna_id] = vec

        # co-fold features for interactions
        con_vecs = {}
        size = args.sirna_len + args.mrna_len + 2
        for row in df.itertuples(index=False):
            inter_id = f"{row.siRNA}_{row.mRNA}"
            pid = _safe_id(inter_id)
            seq_pair = f"{row.siRNA_seq}&{row.mRNA_seq}"
            dp = _run_rnacofold(seq_pair, pid, tmp)
            mat = _parse_dp_ps(dp, size)
            mask = np.ones(size, dtype=bool)
            mask[args.sirna_len:args.sirna_len + 2] = False
            mat = mat[mask][:, mask]
            vec = _reduce_matrix(mat, args.con_svd)
            con_vecs[inter_id] = vec

    con_df = pd.DataFrame.from_dict(con_vecs, orient="index")
    sirna_df = pd.DataFrame.from_dict(sirna_vecs, orient="index")
    mrna_df = pd.DataFrame.from_dict(mrna_vecs, orient="index")

    con_df.to_csv(os.path.join(args.output_dir, "con_matrix.txt"), header=False)
    sirna_df.to_csv(os.path.join(args.output_dir, "self_siRNA_matrix.txt"), header=False)
    mrna_df.to_csv(os.path.join(args.output_dir, "self_mRNA_matrix.txt"), header=False)


if __name__ == "__main__":
    main()
