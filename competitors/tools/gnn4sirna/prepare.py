#!/usr/bin/env python
import argparse
import hashlib
import os
import shutil
import subprocess
import sys

import pandas as pd


def write_fasta(path, rows):
    with open(path, 'w') as f:
        for rid, seq in rows:
            f.write(f">{rid}\n{seq}\n")


def revcomp(seq):
    comp = str.maketrans("ATCGU", "TAGCA")
    return seq.upper().translate(comp)[::-1]


def stable_id(prefix, seq):
    digest = hashlib.md5(seq.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def run_preprocess(preprocess_src, raw_dir, processed_dir):
    work_dir = os.path.join(processed_dir, "_preprocess_work")
    os.makedirs(work_dir, exist_ok=True)
    for fname in os.listdir(preprocess_src):
        if fname.endswith('.py') or fname == 'params.py':
            shutil.copy(os.path.join(preprocess_src, fname), os.path.join(work_dir, fname))

    params_path = os.path.join(work_dir, "params.py")
    with open(params_path, 'w') as f:
        f.write("sirna_fasta_file = '../raw/sirna.fas'\n")
        f.write("mrna_fasta_file = '../raw/mRNA.fas'\n")
        f.write("sirna_mrna_efficacy = '../raw/sirna_mrna_efficacy.csv'\n")
        f.write("k_sirna = 3\n")
        f.write("k_mrna = 4\n")

    shutil.copytree(raw_dir, os.path.join(processed_dir, "raw"), dirs_exist_ok=True)

    python_exec = sys.executable or "python3"

    def run(cmd):
        subprocess.check_call([python_exec] + cmd, cwd=work_dir)

    run(["1_make_kmer.py"])
    run(["2_make_thermo_feature.py"])

    # prepare RNAup input
    eff_path = os.path.join(raw_dir, "sirna_mrna_efficacy.csv")
    eff = pd.read_csv(eff_path)
    datase_tofold = []
    for _, row in eff.iterrows():
        sirna_id = row["siRNA"]
        mrna_id = row["mRNA"]
        sirna_seq = row["siRNA_seq"].upper().replace('U', 'T')
        mrna_seq = row["mRNA_seq"].upper().replace('U', 'T')
        datase_tofold.append([sirna_id, revcomp(sirna_seq), mrna_id, mrna_seq])
    pd.DataFrame(datase_tofold).to_csv(os.path.join(work_dir, "datase_tofold.csv"), index=False, header=False)

    run(["4_make_RNAUp.py"])
    run(["5_make_thermo_profile.py"])

    os.makedirs(processed_dir, exist_ok=True)
    for name, src_name in [
        ("sirna_kmers.txt", "sirna_kmers.txt"),
        ("target_kmers.txt", "mRNA_kmers.txt"),
        ("sirna_target_thermo.csv", "sirna_target_thermo.csv"),
    ]:
        src_path = os.path.join(work_dir, src_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(processed_dir, name))

    shutil.rmtree(work_dir)


def processed_ready(processed_dir):
    if not os.path.isdir(processed_dir):
        return False
    required = [
        os.path.join(processed_dir, "sirna_kmers.txt"),
        os.path.join(processed_dir, "sirna_target_thermo.csv"),
    ]
    for path in required:
        if not os.path.exists(path):
            return False
    target_kmers = os.path.join(processed_dir, "target_kmers.txt")
    mrna_kmers = os.path.join(processed_dir, "mRNA_kmers.txt")
    return os.path.exists(target_kmers) or os.path.exists(mrna_kmers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", default="data")
    p.add_argument("--dataset-name")
    p.add_argument("--id-col", default="id")
    p.add_argument("--sirna-col", default="siRNA")
    p.add_argument("--mrna-col", default="extended_mRNA")
    p.add_argument("--efficacy-col", default="efficacy")
    p.add_argument("--run-preprocess", action="store_true")
    p.add_argument("--gnn-src", default="gnn4sirna_src")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    dataset_name = args.dataset_name or os.path.splitext(os.path.basename(args.input_csv))[0]

    if args.id_col not in df.columns:
        df[args.id_col] = [f"row_{i}" for i in range(len(df))]

    df[args.sirna_col] = df[args.sirna_col].astype(str).str.upper().str.replace('U', 'T')
    df[args.mrna_col] = df[args.mrna_col].astype(str).str.upper().str.replace('U', 'T')

    df["sirna_id"] = [stable_id("sirna", seq) for seq in df[args.sirna_col]]
    df["mrna_id"] = [stable_id("mrna", seq) for seq in df[args.mrna_col]]

    out_df = pd.DataFrame({
        "id": df[args.id_col],
        "siRNA": df["sirna_id"],
        "mRNA": df["mrna_id"],
        "efficacy": df[args.efficacy_col].astype(float),
        "siRNA_seq": df[args.sirna_col],
        "mRNA_seq": df[args.mrna_col],
    })

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"{dataset_name}.csv")
    out_df.to_csv(out_csv, index=False)

    raw_dir = os.path.join(args.output_dir, "raw", dataset_name)
    os.makedirs(raw_dir, exist_ok=True)
    sirna_fa = os.path.join(raw_dir, "sirna.fas")
    mrna_fa = os.path.join(raw_dir, "mRNA.fas")
    write_fasta(sirna_fa, zip(out_df["siRNA"], out_df["siRNA_seq"]))
    write_fasta(mrna_fa, zip(out_df["mRNA"], out_df["mRNA_seq"]))

    eff_csv = os.path.join(raw_dir, "sirna_mrna_efficacy.csv")
    out_df[["siRNA", "mRNA", "efficacy", "siRNA_seq", "mRNA_seq"]].to_csv(eff_csv, index=False)

    processed_dir = os.path.join(args.output_dir, "processed", dataset_name)
    if args.run_preprocess or not processed_ready(processed_dir):
        preprocess_src = os.path.abspath(os.path.join(os.path.dirname(__file__), args.gnn_src, "preprocessing"))
        if not os.path.isdir(preprocess_src):
            raise FileNotFoundError(f"preprocessing not found: {preprocess_src}")
        run_preprocess(preprocess_src, raw_dir, processed_dir)

    print(out_csv)


if __name__ == "__main__":
    main()
