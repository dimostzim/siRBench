#!/usr/bin/env python3

"""
Build a comprehensive siRNA–target feature dataset by combining
thermodynamic features from folder 1 and folder 2 without duplicates.

- Keeps folder 2 "oligoformer" features, excluding DG_1/DG_2/DG_13/DG_18
  since they duplicate the per-position ΔG vector we add from folder 1.
- Adds folder 1 per-dinucleotide ΔG vector (positions 1–18) and total ΔG sum
  computed with folder 1 logic (initiation + symmetry + 5' A and 3' U bonuses).
- Adds RNAup energies (opening, interaction, total) by default and RNAcofold MFE
  (both optional at runtime but enabled unless explicitly disabled).

Inputs: CSV with columns containing "sirna" and "mrna" or "target" in the names.
Outputs: CSV with unified feature columns. By default, standardized copies of
the sequences are not included (use --with-std-seqs to add them).
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import pandas as pd


def _import_features_calculator():
    """Import features_calculator from the `2/` folder by adding it to sys.path."""
    here = os.path.dirname(os.path.abspath(__file__))
    fc_dir = os.path.join(here, '2')
    if fc_dir not in sys.path:
        sys.path.insert(0, fc_dir)
    import features_calculator as fc  # type: ignore
    return fc


def _find_cols(df: pd.DataFrame) -> Tuple[str, str]:
    sirna_col = None
    target_col = None
    for col in df.columns:
        cl = str(col).lower()
        if sirna_col is None and 'sirna' in cl:
            sirna_col = col
        if target_col is None and ('mrna' in cl or 'target' in cl):
            target_col = col
    if sirna_col is None or target_col is None:
        raise ValueError("Could not find siRNA and target/mRNA columns in the input file.")
    return sirna_col, target_col


def _std_seq(s: str, length: int = 19) -> str:
    s = (s or '').strip().upper().replace('T', 'U')
    # Folder 1 logic uses the first 19 nts
    return s[:length]


def compute_dg_vector_and_sum_folder1(seq: str, dg_map: dict) -> Tuple[List[float], float]:
    """Compute per-dinucleotide ΔG vector (positions 1–18) and total ΔG with folder 1 logic.

    Folder 1 total ΔG logic:
    - Start with initiation +4.09 and symmetry correction +0.43
    - Add +0.45 if first base is 'A'
    - Add +0.45 if last base (position 19) is 'U'
    - Add nearest-neighbor ΔG for each dinucleotide (positions 1..18)
    """
    if len(seq) < 19:
        raise ValueError("siRNA sequence must be at least 19 nt long after standardization.")

    intermolecular_initiation = 4.09
    symmetry_correction = 0.43
    end_bonus = 0.45

    # Per-position ΔG for dinucleotides
    dg_vec: List[float] = []
    for i in range(18):
        dinuc = seq[i:i + 2]
        dg_vec.append(float(dg_map.get(dinuc, 0.0)))

    total = 0.0
    # End effects per folder 1
    if seq[0] == 'A':
        total += end_bonus
    if seq[18] == 'U':  # position 19
        total += end_bonus

    total += sum(dg_vec)
    total += intermolecular_initiation
    total += symmetry_correction

    return dg_vec, float(f"{total:.2f}")


def try_rnaup_energies(sirna_seq: str, target_seq: str) -> Optional[Tuple[float, float, float]]:
    """Compute RNAup energies (opening, interaction, total) for siRNA-target pair.

    Returns a tuple (opening_dG, interaction_dG, total_dG) or None if RNAup is unavailable
    or parsing fails. Designed to be optional.
    """
    import subprocess

    # Build input: two sequences, each on its own line
    input_data = f"{sirna_seq}\n{target_seq}\n"
    try:
        # Using settings similar to folder 1; omit output redirection file argument
        proc = subprocess.run(
            ["RNAup", "-b", "-d2", "--noLP", "-c", "S"],
            input=input_data,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    except Exception:
        return None

    out = proc.stdout or ''
    # Heuristic parse: look for the last parenthesized triple; extract numbers
    # Example patterns often include lines like:  "... ( -X = -Y + -Z )"
    line = None
    for l in out.strip().splitlines()[::-1]:
        if '(' in l and ')' in l and '=' in l:
            line = l
            break
    if not line:
        return None

    try:
        # Extract inside parentheses and split numbers
        start = line.rfind('(')
        end = line.rfind(')')
        inner = line[start + 1:end]
        # Remove symbols and split
        cleaned = inner.replace('=', ' ').replace('+', ' ')
        parts = [p for p in cleaned.split() if p]
        # Usually order is: total, opening, interaction; reorder to (opening, interaction, total)
        vals = [float(x) for x in parts]
        if len(vals) >= 3:
            total = vals[0]
            opening = vals[1]
            interaction = vals[2]
            return float(opening), float(interaction), float(total)
    except Exception:
        return None
    return None


def build_unified_features(
    df: pd.DataFrame,
    include_rnaup: bool = True,
    include_cofold: bool = True,
    include_std_seqs: bool = False,
) -> pd.DataFrame:
    fc = _import_features_calculator()
    sirna_col, target_col = _find_cols(df)

    # Prepare outputs
    out_rows = []

    # Names for folder 2 features; we will drop DG_1/DG_2/DG_13/DG_18 to avoid duplicates
    f2_names = fc.get_feature_names()
    duplicate_names = {"DG_1", "DG_2", "DG_13", "DG_18"}

    # Canonical names for folder 1 vector and sum
    dg_vec_names = [f"DG_pos{i}" for i in range(1, 19)]
    dg_sum_name = "DG_sum_f1"

    # Optional energy names
    rnaup_names = ["RNAup_open_dG", "RNAup_interaction_dG", "RNAup_total_dG"]
    cofold_name = "duplex_cofold_dG"

    for _, row in df.iterrows():
        sirna_seq = _std_seq(str(row[sirna_col]))
        target_seq = _std_seq(str(row[target_col]))

        # Folder 2 features
        f2_vals = fc.calculate_oligoformer_features_exact(sirna_seq)
        f2_dict = {name: val for name, val in zip(f2_names, f2_vals)}
        # Drop duplicates
        for k in list(f2_dict.keys()):
            if k in duplicate_names:
                del f2_dict[k]

        # Folder 1 vector and sum (using same ΔG map as folder 2 to ensure consistency)
        dg_vec, dg_sum = compute_dg_vector_and_sum_folder1(sirna_seq, fc.DELTA_G)
        f1_dict = {dg_sum_name: dg_sum}
        f1_dict.update({name: val for name, val in zip(dg_vec_names, dg_vec)})

        # Optional energies
        energy_dict = {}
        if include_cofold:
            mfe = fc.calculate_duplex_folding_energy(sirna_seq, target_seq)
            energy_dict[cofold_name] = round(mfe, 2) if mfe is not None else None
        if include_rnaup:
            rnaup = try_rnaup_energies(sirna_seq, target_seq)
            if rnaup is None:
                energy_dict.update({k: None for k in rnaup_names})
            else:
                energy_dict[rnaup_names[0]] = round(rnaup[0], 2)
                energy_dict[rnaup_names[1]] = round(rnaup[1], 2)
                energy_dict[rnaup_names[2]] = round(rnaup[2], 2)

        # Compose final row
        base = {}
        if include_std_seqs:
            base['sirna_seq'] = sirna_seq
            base['target_seq'] = target_seq
        out_rows.append({**base, **f1_dict, **f2_dict, **energy_dict})

    out_df = pd.DataFrame(out_rows)
    return pd.concat([df.reset_index(drop=True), out_df], axis=1)


def main():
    p = argparse.ArgumentParser(description='Build unified siRNA–target feature dataset without duplicates.')
    p.add_argument('input', help='Input CSV with siRNA and target/mRNA columns')
    p.add_argument('-o', '--output', required=True, help='Output CSV file')
    p.add_argument('--no-rnaup', action='store_true', help='Disable RNAup energies')
    p.add_argument('--no-cofold', action='store_true', help='Disable RNAcofold MFE calculation')
    p.add_argument('--with-std-seqs', action='store_true', help='Include standardized siRNA/target sequences in output')
    args = p.parse_args()

    df = pd.read_csv(args.input)
    out = build_unified_features(
        df,
        include_rnaup=not args.no_rnaup,
        include_cofold=not args.no_cofold,
        include_std_seqs=args.with_std_seqs,
    )
    out.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
