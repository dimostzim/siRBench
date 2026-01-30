#!/usr/bin/env python3

"""
Build a siRNA–target feature dataset with robust feature selection.

Combines the following feature groups:
- Sequence composition features (oligoformer): U_all, G_all, dinucleotide counts, etc.
- Advanced energy calculations: Per-base folding effects via constraints (RNAfold/RNAcofold)
- RNA interaction energies: RNAup (opening and interaction)

This replaces earlier placeholder mutations with proper constraint-based analyses and
removes RNA-FM embedding features entirely for portability and speed.

Inputs: CSV with columns containing "siRNA" and "mRNA".
Outputs: CSV with a feature set optimized for siRNA efficacy prediction.
"""

import argparse
import os
import sys
import subprocess
import re
from functools import lru_cache
from typing import List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_calculator as fc


def _std_seq(s: str, length: int = 19) -> str:
    """Standardize sequence to uppercase U notation, truncate to length."""
    s = (s or '').strip().upper().replace('T', 'U')
    return s[:length]


def _parse_energy_from_output(text: str) -> Optional[float]:
    """Parse MFE value from RNAfold/cofold output lines like "... ( -3.40 )"."""
    matches = re.findall(r"\(([^()]+)\)", text)
    for m in reversed(matches):
        try:
            return float(m.strip())
        except Exception:
            continue
    return None


@lru_cache(maxsize=10000)
def _rnafold_mfe(seq: str) -> float:
    proc = subprocess.run(
        ["RNAfold", "--noPS"],
        input=f"{seq}\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"RNAfold failed: {proc.stderr}")
    energy = _parse_energy_from_output(proc.stdout)
    if energy is None:
        raise RuntimeError("RNAfold did not produce an energy value")
    return energy


@lru_cache(maxsize=100000)
def _rnafold_mfe_with_unpaired(seq: str, pos: int) -> float:
    if pos < 0 or pos >= len(seq):
        raise ValueError("Position out of range for RNAfold constraint")
    constraint = list('.' * len(seq))
    constraint[pos] = 'x'
    constraint = ''.join(constraint)
    proc = subprocess.run(
        ["RNAfold", "--noPS", "-C"],
        input=f"{seq}\n{constraint}\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"RNAfold (constraint) failed: {proc.stderr}")
    energy = _parse_energy_from_output(proc.stdout)
    if energy is None:
        raise RuntimeError("RNAfold (constraint) did not produce an energy value")
    return energy


def calculate_single_energy_contributions(sequence: str) -> List[float]:
    """Per-base folding effects via constraints.

    Returns [total MFE, contrib_pos1..contrib_pos19], where contribution at i is
    (E_with_pos_i_unpaired - E_baseline). More positive values mean that enforcing
    an unpaired base destabilizes the native structure.
    """
    seq = sequence.replace('\n', '').replace('T', 'U')
    base = _rnafold_mfe(seq)
    out = [float(base)]
    for i in range(len(seq)):
        ei = _rnafold_mfe_with_unpaired(seq, i)
        out.append(float(ei - base))
    return out


@lru_cache(maxsize=10000)
def _rnacofold_mfe(seq1: str, seq2: str) -> float:
    """Unconstrained cofold MFE via RNAcofold."""
    combined = f"{seq1}&{seq2}"
    proc = subprocess.run(
        ["RNAcofold", "--noPS"],
        input=f"{combined}\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"RNAcofold failed: {proc.stderr}")
    energy = _parse_energy_from_output(proc.stdout)
    if energy is None:
        raise RuntimeError("RNAcofold did not produce an energy value")
    return energy


@lru_cache(maxsize=100000)
def _rnacofold_mfe_with_unpaired(seq1: str, seq2: str, idx: int, which: str) -> float:
    if which not in {"sirna", "target"}:
        raise ValueError("Parameter 'which' must be 'sirna' or 'target'")
    if which == "sirna" and not (0 <= idx < len(seq1)):
        raise ValueError("siRNA index out of range for RNAcofold constraint")
    if which == "target" and not (0 <= idx < len(seq2)):
        raise ValueError("Target index out of range for RNAcofold constraint")
    cons = list('.' * len(seq1) + '&' + '.' * len(seq2))
    if which == 'sirna':
        cons[idx] = 'x'
    else:
        cons[len(seq1) + 1 + idx] = 'x'
    cons_str = ''.join(cons)
    combined = f"{seq1}&{seq2}"
    proc = subprocess.run(
        ["RNAcofold", "--noPS", "-C"],
        input=f"{combined}\n{cons_str}\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"RNAcofold (constraint) failed: {proc.stderr}")
    energy = _parse_energy_from_output(proc.stdout)
    if energy is None:
        raise RuntimeError("RNAcofold (constraint) did not produce an energy value")
    return energy


def calculate_duplex_energy_contributions(seq1: str, seq2: str) -> List[float]:
    """Duplex per-base effects via RNAcofold constraints.

    Returns [total_energy, contrib_sirna_pos1..pos19, contrib_target_pos1..pos19].
    Contribution definition matches single-strand case.
    """
    base = _rnacofold_mfe(seq1, seq2)
    out = [float(base)]
    for i in range(len(seq1)):
        ei = _rnacofold_mfe_with_unpaired(seq1, seq2, i, 'sirna')
        out.append(float(ei - base))
    for j in range(len(seq2)):
        ej = _rnacofold_mfe_with_unpaired(seq1, seq2, j, 'target')
        out.append(float(ej - base))
    return out


def try_rnaup_energies(sirna_seq: str, target_seq: str) -> Tuple[float, float, float]:
    """Compute RNAup energies (opening, interaction, total) for siRNA-target pair.

    Returns a tuple (opening_dG, interaction_dG, total_dG).
    Raises exception if RNAup is unavailable or parsing fails.
    """
    input_data = f"{sirna_seq}\n{target_seq}\n"

    proc = subprocess.run(
        ["RNAup", "-b", "-d2", "--noLP", "-c", "S"],
        input=input_data,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"RNAup failed: {proc.stderr}")

    out = proc.stdout or ''
    summary = None
    for l in out.strip().splitlines()[::-1]:
        if '(' in l and ')' in l and '=' in l and '+' in l:
            summary = l[l.rfind('(')+1:l.rfind(')')]
            break
    if not summary:
        raise ValueError("Could not parse RNAup output")
    nums = re.findall(r"[+-]?(?:\d+\.\d+|\d+|\.\d+)", summary)
    vals = [float(x) for x in nums]
    if len(vals) >= 4:
        total, open1, open2, interaction = vals[0], vals[1], vals[2], vals[3]
        opening = open1 + open2
    elif len(vals) == 3:
        total, opening, interaction = vals
    else:
        raise ValueError("Unexpected RNAup energy format")
    return (float(opening), float(interaction), float(total))


def build_unified_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build siRNA–target feature dataset."""
    sirna_col, target_col = 'siRNA', 'mRNA'

    out_rows = []

    f2_names = fc.get_feature_names()
    include_names = set(f2_names)

    rnaup_names = ["RNAup_open_dG", "RNAup_interaction_dG"]

    print(f"Processing {len(df)} sequences...")
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(df)}")

        sirna_seq = _std_seq(str(row[sirna_col]))
        target_seq = _std_seq(str(row[target_col]))

        f2_vals = fc.calculate_oligoformer_features_exact(sirna_seq)
        comp_dict = {name: val for name, val in zip(f2_names, f2_vals) if name in include_names}

        energy_dict = {}
        single_energies = calculate_single_energy_contributions(sirna_seq)
        for i, energy in enumerate(single_energies):
            if i == 0:
                energy_dict["single_energy_total"] = energy
            else:
                energy_dict[f"single_energy_pos{i}"] = energy

        duplex_energies = calculate_duplex_energy_contributions(sirna_seq, target_seq)
        for i, energy in enumerate(duplex_energies):
            if i == 0:
                energy_dict["duplex_energy_total"] = energy
            elif i <= 19:
                energy_dict[f"duplex_energy_sirna_pos{i}"] = energy
            else:
                pos = i - 19
                energy_dict[f"duplex_energy_target_pos{pos}"] = energy

        rnaup = try_rnaup_energies(sirna_seq, target_seq)
        rna_tool_dict = {
            rnaup_names[0]: rnaup[0],
            rnaup_names[1]: rnaup[1],
        }

        out_rows.append({**comp_dict, **energy_dict, **rna_tool_dict})

    out_df = pd.DataFrame(out_rows)
    return pd.concat([df.reset_index(drop=True), out_df], axis=1)


def main():
    p = argparse.ArgumentParser(description='Build siRNA–target feature dataset.')
    p.add_argument('input', nargs='?', default='siRBench_base.csv', help='Input CSV with siRNA and target/mRNA columns (default: siRBench_base.csv)')
    p.add_argument('-o', '--output', help='Output CSV file (default: siRBench_with_features.csv)')
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, '..'))

    if not args.output:
        if args.input == 'siRBench_base.csv':
            args.output = os.path.join(data_dir, 'siRBench_with_features.csv')
        else:
            base_name = os.path.basename(args.input).rsplit('.', 1)[0]
            args.output = os.path.join(data_dir, f'{base_name}_with_features.csv')

    input_path = args.input
    if not os.path.isabs(input_path):
        if not os.path.exists(input_path):
            candidate = os.path.join(data_dir, input_path)
            if os.path.exists(candidate):
                input_path = candidate

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} sequences")
    print(f"Columns: {list(df.columns)}")

    if 'siRNA' in df.columns and 'mRNA' in df.columns:
        print("\nSample sequences:")
        for i in range(min(3, len(df))):
            sirna = df.iloc[i]['siRNA']
            mrna = df.iloc[i]['mRNA']
            efficacy = df.iloc[i].get('efficacy', 'N/A')
            print(f"  {i+1}: siRNA={sirna[:20]}... mRNA={mrna[:20]}... efficacy={efficacy}")

    print("\nBuilding feature set...")
    out = build_unified_features(df)

    output_path = args.output

    print(f"\nSaving {out.shape[1]} features to {output_path}...")
    float_cols = out.select_dtypes(include=['float64', 'float32']).columns
    out[float_cols] = out[float_cols].round(3)
    out.to_csv(output_path, index=False)
    print(f"Enhanced dataset saved with {out.shape[0]} sequences and {out.shape[1]} features")
    print("Done!")


if __name__ == '__main__':
    main()
