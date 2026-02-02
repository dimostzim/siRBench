#!/usr/bin/env python
import argparse
import os
import sys
import subprocess

import pandas as pd


DeltaG = {
    'AA': -0.93, 'UU': -0.93, 'AU': -1.10, 'UA': -1.33, 'CU': -2.08,
    'AG': -2.08, 'CA': -2.11, 'UG': -2.11, 'GU': -2.24, 'AC': -2.24,
    'GA': -2.35, 'UC': -2.35, 'CG': -2.36, 'GG': -3.26, 'CC': -3.26,
    'GC': -3.42, 'init': 4.09, 'endAU': 0.45, 'sym': 0.43,
}
DeltaH = {
    'AA': -6.82, 'UU': -6.82, 'AU': -9.38, 'UA': -7.69, 'CU': -10.48,
    'AG': -10.48, 'CA': -10.44, 'UG': -10.44, 'GU': -11.40, 'AC': -11.40,
    'GA': -12.44, 'UC': -12.44, 'CG': -10.64, 'GG': -13.39, 'CC': -13.39,
    'GC': -14.88, 'init': 3.61, 'endAU': 3.72, 'sym': 0,
}


def anti_rna(rna):
    comp = {'A': 'U', 'U': 'A', 'T': 'A', 'C': 'G', 'G': 'C', 'X': 'X'}
    return ''.join(comp.get(b, 'X') for b in rna.upper())[::-1]


def calculate_dgh(seq):
    dg_all = 0
    dg_all += DeltaG['init']
    dg_all += ((seq[0] + seq[len(seq)-1]).count('A') + (seq[0] + seq[len(seq)-1]).count('U')) * DeltaG['endAU']
    dg_all += DeltaG['sym'] if anti_rna(seq).replace('T','U') == seq else 0
    for i in range(len(seq) - 1):
        dg_all += DeltaG[seq[i] + seq[i+1]]
    dh_all = 0
    dh_all += DeltaH['init']
    dh_all += ((seq[0] + seq[len(seq)-1]).count('A') + (seq[0] + seq[len(seq)-1]).count('U')) * DeltaH['endAU']
    dh_all += DeltaH['sym'] if anti_rna(seq).replace('T','U') == seq else 0
    for i in range(len(seq) - 1):
        dh_all += DeltaH[seq[i] + seq[i+1]]
    return dg_all, dh_all


def calculate_end_diff(sirna):
    count = 0
    _5 = sirna[:2]
    _3 = sirna[-2:]
    if _5 in ['AC','AG','UC','UG']:
        count += 1
    elif _5 in ['GA','GU','CA','CU']:
        count -= 1
    if _3 in ['AC','AG','UC','UG']:
        count += 1
    elif _3 in ['GA','GU','CA','CU']:
        count -= 1
    return float('{:.2f}'.format(DeltaG[_5] - DeltaG[_3] + count * 0.45))


def td_features(seq):
    seq = seq.upper().replace('T', 'U')
    if len(seq) < 19:
        raise ValueError(f"siRNA sequence too short for td features: {seq}")
    seq = seq[:19]
    if any(b not in "AUCG" for b in seq):
        raise ValueError(f"siRNA sequence contains non-AUCG bases: {seq}")
    dg_all, dh_all = calculate_dgh(seq)
    pairs = [seq[i:i+2] for i in range(18)]
    return [
        calculate_end_diff(seq),
        DeltaG[seq[0:2]],
        DeltaH[seq[0:2]],
        int(seq[0] == 'U'),
        int(seq[0] == 'G'),
        dh_all,
        seq.count('U') / 19,
        int(seq[0:2] == 'UU'),
        seq.count('G') / 19,
        int(seq[0:2] == 'GG'),
        int(seq[0:2] == 'GC'),
        pairs.count('GG') / 18,
        DeltaG[seq[1:3]],
        pairs.count('UA') / 18,
        int(seq[1] == 'U'),
        int(seq[0] == 'C'),
        pairs.count('CC') / 18,
        DeltaG[seq[17:19]],
        int(seq[0:2] == 'CC'),
        pairs.count('GC') / 18,
        int(seq[0:2] == 'CG'),
        DeltaG[seq[12:14]],
        pairs.count('UU') / 18,
        int(seq[18] == 'A'),
    ]


def write_fasta(path, rows):
    with open(path, 'w') as f:
        for rid, seq in rows:
            f.write(f">{rid}\n{seq}\n")


def run_rnafm(rnafm_root, fasta_path, out_dir):
    workdir = os.path.join(rnafm_root, "redevelop")
    cmd = [
        sys.executable,
        "launch/predict.py",
        "--config=pretrained/extract_embedding.yml",
        f"--data_path={fasta_path}",
        f"--save_dir={out_dir}",
        "--save_frequency", "1",
        "--save_embeddings",
    ]
    subprocess.check_call(cmd, cwd=workdir)


def rnafm_ready(output_dir, dataset_name):
    base = os.path.join(output_dir, "RNAFM")
    mrna_dir = os.path.join(base, f"{dataset_name}_mRNA", "representations")
    sirna_dir = os.path.join(base, f"{dataset_name}_siRNA", "representations")
    return os.path.isdir(mrna_dir) and os.path.isdir(sirna_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", default="data")
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--id-col", default="id")
    p.add_argument("--sirna-col", default="siRNA")
    p.add_argument("--mrna-col", default="extended_mRNA")
    p.add_argument("--efficiency-col", default="efficiency")
    p.add_argument("--binary-col", default="binary")
    p.add_argument("--binary-threshold", type=float, default=0.7)
    p.add_argument("--run-rnafm", action="store_true")
    p.add_argument("--rnafm-root", default="oligoformer_src/RNA-FM")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)

    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        base = os.path.basename(args.input_csv)
        dataset_name = os.path.splitext(base)[0]

    if args.id_col not in df.columns:
        df[args.id_col] = [f"row_{i}" for i in range(len(df))]

    df[args.sirna_col] = df[args.sirna_col].astype(str).str.upper().str.replace('T', 'U')
    df[args.mrna_col] = df[args.mrna_col].astype(str).str.upper().str.replace('T', 'U')
    df[args.mrna_col] = df[args.mrna_col].str.replace('N', 'X')

    td_vals = [td_features(seq) for seq in df[args.sirna_col]]
    td_str = [','.join(str(x) for x in row) for row in td_vals]

    if args.binary_col in df.columns:
        y_vals = df[args.binary_col].astype(int)
    else:
        y_vals = (df[args.efficiency_col].astype(float) >= args.binary_threshold).astype(int)

    out_df = pd.DataFrame({
        "id": df[args.id_col],
        "siRNA": df[args.sirna_col],
        "mRNA": df[args.mrna_col],
        "label": df[args.efficiency_col].astype(float),
        "y": y_vals,
        "td": td_str,
    })

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"{dataset_name}.csv")
    out_df.to_csv(csv_path, index=False)

    fasta_dir = os.path.join(args.output_dir, "fasta")
    os.makedirs(fasta_dir, exist_ok=True)
    sirna_fa = os.path.join(fasta_dir, f"{dataset_name}_siRNA.fa")
    mrna_fa = os.path.join(fasta_dir, f"{dataset_name}_mRNA.fa")
    write_fasta(sirna_fa, zip(out_df["id"], out_df["siRNA"]))
    write_fasta(mrna_fa, zip(out_df["id"], out_df["mRNA"]))

    if args.run_rnafm or not rnafm_ready(args.output_dir, dataset_name):
        rnafm_root = os.path.abspath(args.rnafm_root)
        if not os.path.isdir(rnafm_root):
            raise FileNotFoundError(f"RNA-FM not found: {rnafm_root}")
        rnafm_dir = os.path.join(args.output_dir, "RNAFM")
        os.makedirs(rnafm_dir, exist_ok=True)
        run_rnafm(rnafm_root, mrna_fa, os.path.join(rnafm_dir, f"{dataset_name}_mRNA"))
        run_rnafm(rnafm_root, sirna_fa, os.path.join(rnafm_dir, f"{dataset_name}_siRNA"))

    print(csv_path)


if __name__ == "__main__":
    main()
