#!/usr/bin/env python
import argparse
import json
import os
import re
import subprocess
import sys

import pandas as pd


def revcomp(seq):
    comp = str.maketrans("AUGC", "UACG")
    return seq.upper().replace('T', 'U').translate(comp)[::-1]


def clean_seq(seq):
    seq = str(seq).upper().replace('T', 'U')
    return seq.replace('X', '').replace('N', '')


def resolve_position(mrna_seq, sense_seq, anti_seq, target_seq=None):
    if target_seq and target_seq in mrna_seq:
        return mrna_seq.index(target_seq)
    if sense_seq in mrna_seq:
        return mrna_seq.index(sense_seq)
    if anti_seq in mrna_seq:
        return mrna_seq.index(anti_seq)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--pdb-dir", default=None)
    p.add_argument("--rosetta-dir", default=None)
    p.add_argument("--id-col", default="id")
    p.add_argument("--sirna-col", default="siRNA")
    p.add_argument("--mrna-col", default="extended_mRNA")
    p.add_argument("--efficiency-col", default="efficiency")
    p.add_argument("--pdb-path-col", default="pdb_data_path")
    p.add_argument("--chain-col", default="chain")
    p.add_argument("--start-col", default="start")
    p.add_argument("--position-col", default="position")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.id_col not in df.columns:
        df[args.id_col] = [f"row_{i}" for i in range(len(df))]

    safe_ids = df[args.id_col].astype(str).apply(lambda x: re.sub(r"[^A-Za-z0-9_.-]", "_", x))
    if safe_ids.duplicated().any():
        safe_ids = safe_ids + "_" + df.index.astype(str)
    df["pdb_id"] = safe_ids

    pdb_col = args.pdb_path_col
    have_pdb_col = pdb_col in df.columns
    needs_pdb = (~have_pdb_col) or df[pdb_col].isna().any()

    pdb_dir = args.pdb_dir
    if pdb_dir is None:
        pdb_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_jsonl)), "pdb")

    if needs_pdb:
        rosetta_dir = args.rosetta_dir or os.environ.get("ROSETTA_DIR")
        if not rosetta_dir:
            raise ValueError("Missing PDBs and ROSETTA_DIR not set; provide pdb_data_path or set --rosetta-dir/ROSETTA_DIR.")

        os.makedirs(pdb_dir, exist_ok=True)

        df_pdb = df.copy()
        df_pdb["sirna_seq"] = df_pdb[args.sirna_col].astype(str).str.upper().str.replace('T', 'U')
        df_pdb["sense seq"] = df_pdb["sirna_seq"].apply(revcomp)
        df_pdb["anti seq"] = df_pdb["sirna_seq"]
        df_pdb["mRNA_seq"] = df_pdb[args.mrna_col].apply(clean_seq)
        if "mRNA" in df.columns:
            df_pdb["target_seq"] = df_pdb["mRNA"].apply(clean_seq)
        else:
            df_pdb["target_seq"] = None

        def find_position(row):
            if args.position_col in df.columns and not pd.isna(row[args.position_col]):
                return int(row[args.position_col])
            return int(resolve_position(row["mRNA_seq"], row["sense seq"], row["anti seq"], row.get("target_seq")))

        df_pdb["position"] = df_pdb.apply(find_position, axis=1)

        pdb_input = df_pdb[df_pdb[pdb_col].isna() if have_pdb_col else df_pdb.index == df_pdb.index]
        pdb_csv = os.path.join(os.path.dirname(os.path.abspath(args.output_jsonl)), "pdb_input.csv")
        pdb_input[["pdb_id", "mRNA_seq", "position", "sense seq", "anti seq", args.efficiency_col]].rename(
            columns={"pdb_id": "siRNA", args.efficiency_col: "efficiency"}
        ).to_csv(pdb_csv, index=False)

        src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "ensirna_src", "ENsiRNA"))
        cmd = [
            sys.executable,
            "-m",
            "data.get_pdb",
            "-f",
            pdb_csv,
            "-p",
            pdb_dir,
            "--rosetta-dir",
            rosetta_dir,
        ]
        subprocess.check_call(cmd, cwd=src_root)

        pdb_json = os.path.splitext(pdb_csv)[0] + ".json"
        if not os.path.exists(pdb_json):
            raise FileNotFoundError(f"Missing PDB metadata: {pdb_json}")
        pdb_df = pd.read_json(pdb_json, lines=True)
        pdb_df = pdb_df.set_index("siRNA")

        df[pdb_col] = df.get(pdb_col)
        df[pdb_col] = df[pdb_col].where(~df[pdb_col].isna(), df["pdb_id"].map(pdb_df["pdb_data_path"]))
        if args.start_col in df.columns:
            df[args.start_col] = df[args.start_col].where(~df[args.start_col].isna(), df["pdb_id"].map(pdb_df["start"]))
        else:
            df[args.start_col] = df["pdb_id"].map(pdb_df["start"])
        if args.chain_col in df.columns:
            df[args.chain_col] = df[args.chain_col].where(~df[args.chain_col].isna(), df["pdb_id"].map(pdb_df["chain"]))
        else:
            df[args.chain_col] = df["pdb_id"].map(pdb_df["chain"])

        df = df.dropna(subset=[pdb_col, args.start_col, args.chain_col])

    with open(args.output_jsonl, 'w') as f:
        for _, row in df.iterrows():
            sirna = str(row[args.sirna_col]).upper().replace('T', 'U')
            anti_seq = sirna
            sense_seq = revcomp(sirna)

            mrna_seq = clean_seq(row[args.mrna_col])
            target_seq = clean_seq(row["mRNA"]) if "mRNA" in df.columns else None

            position = row[args.position_col] if args.position_col in df.columns else None
            if position is None or pd.isna(position):
                position = resolve_position(mrna_seq, sense_seq, anti_seq, target_seq)

            pdb_path = row[pdb_col] if pdb_col in df.columns else None
            if pdb_path is None or pd.isna(pdb_path):
                raise ValueError("pdb_data_path is required for ENsiRNA")

            item = {
                "id": row[args.id_col],
                "pdb": os.path.basename(str(pdb_path)),
                "pdb_data_path": str(pdb_path),
                "chain": row[args.chain_col],
                "start": row[args.start_col],
                "position": int(position),
                "mRNA_seq": mrna_seq,
                "sense seq": sense_seq,
                "anti seq": anti_seq,
                "efficiency": float(row[args.efficiency_col]),
            }
            f.write(json.dumps(item) + "\n")

    print(args.output_jsonl)


if __name__ == "__main__":
    main()
