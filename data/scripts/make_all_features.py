#!/usr/bin/env python3

"""
Build a comprehensive siRNA–target feature dataset with robust feature selection.

Combines the following feature groups:
- Sequence composition features (oligoformer): U_all, G_all, dinucleotide counts, etc.
- Advanced energy calculations: Per-base folding effects via constraints (RNAfold/RNAcofold)
- RNA interaction energies: RNAup (opening and interaction)

This replaces earlier placeholder mutations with proper constraint-based analyses and
removes RNA-FM embedding features entirely for portability and speed.

Inputs: CSV with columns containing "siRNA" and "mRNA".
Outputs: CSV with a comprehensive feature set optimized for siRNA efficacy prediction.
"""

import argparse
import os
import sys
import subprocess
import re
from functools import lru_cache
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np

# Feature calculator import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_calculator as fc


def _find_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """Return exact siRBench column names: 'siRNA' and 'mRNA'."""
    return 'siRNA', 'mRNA'


def _std_seq(s: str, length: int = 19) -> str:
    """Standardize sequence to uppercase U notation, truncate to length."""
    s = (s or '').strip().upper().replace('T', 'U')
    return s[:length]


###########################################################################
# Energy calculations (constraint-based using RNAfold/RNAcofold CLIs)
###########################################################################

_rnafold_missing_warned = False
_rnacofold_missing_warned = False
_rnaup_missing_warned = False


def _parse_energy_from_output(text: str) -> Optional[float]:
    """Parse MFE value from RNAfold/cofold output lines like "... ( -3.40 )"."""
    # Find last parenthesized numeric value
    matches = re.findall(r"\(([^()]+)\)", text)
    for m in reversed(matches):
        try:
            return float(m.strip())
        except Exception:
            continue
    return None


@lru_cache(maxsize=10000)
def _rnafold_mfe(seq: str) -> Optional[float]:
    global _rnafold_missing_warned
    try:
        proc = subprocess.run(
            ["RNAfold", "--noPS"],
            input=f"{seq}\n",
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        if not _rnafold_missing_warned:
            print("WARNING: RNAfold not found. Single-strand energies will be NaN.")
            _rnafold_missing_warned = True
        return None
    if proc.returncode != 0:
        return None
    return _parse_energy_from_output(proc.stdout)


@lru_cache(maxsize=100000)
def _rnafold_mfe_with_unpaired(seq: str, pos: int) -> Optional[float]:
    """MFE with base at index 'pos' constrained to be unpaired (x)."""
    global _rnafold_missing_warned
    if pos < 0 or pos >= len(seq):
        return None
    constraint = list('.' * len(seq))
    constraint[pos] = 'x'
    constraint = ''.join(constraint)
    try:
        proc = subprocess.run(
            ["RNAfold", "--noPS", "-C"],
            input=f"{seq}\n{constraint}\n",
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        if not _rnafold_missing_warned:
            print("WARNING: RNAfold not found. Single-strand energies will be NaN.")
            _rnafold_missing_warned = True
        return None
    if proc.returncode != 0:
        return None
    return _parse_energy_from_output(proc.stdout)


def calculate_single_energy_contributions(sequence: str) -> List[float]:
    """Per-base folding effects via constraints.

    Returns [total MFE, contrib_pos1..contrib_pos19], where contribution at i is
    -(E_with_pos_i_unpaired - E_baseline). More positive => stronger base-specific
    stabilization in the baseline structure.
    """
    seq = sequence.replace('\n', '').replace('T', 'U')
    base = _rnafold_mfe(seq)
    if base is None:
        return [np.nan] + [np.nan] * len(seq)
    out = [float(base)]
    for i in range(len(seq)):
        ei = _rnafold_mfe_with_unpaired(seq, i)
        if ei is None:
            out.append(np.nan)
        else:
            out.append(float(-(ei - base)))
    return out


@lru_cache(maxsize=10000)
def _rnacofold_mfe(seq1: str, seq2: str) -> Optional[float]:
    """Unconstrained cofold MFE via RNAcofold."""
    global _rnacofold_missing_warned
    combined = f"{seq1}&{seq2}"
    try:
        proc = subprocess.run(
            ["RNAcofold", "--noPS"],
            input=f"{combined}\n",
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        if not _rnacofold_missing_warned:
            print("WARNING: RNAcofold not found. Duplex energies will be NaN.")
            _rnacofold_missing_warned = True
        return None
    if proc.returncode != 0:
        return None
    return _parse_energy_from_output(proc.stdout)


@lru_cache(maxsize=100000)
def _rnacofold_mfe_with_unpaired(seq1: str, seq2: str, idx: int, which: str) -> Optional[float]:
    """Cofold MFE with base constrained unpaired.

    which: 'sirna' or 'target', idx is 0-based within that strand.
    """
    global _rnacofold_missing_warned
    if which not in {"sirna", "target"}:
        return None
    if which == "sirna" and not (0 <= idx < len(seq1)):
        return None
    if which == "target" and not (0 <= idx < len(seq2)):
        return None
    # Build constraint string over combined sequence with '&'
    cons = list('.' * len(seq1) + '&' + '.' * len(seq2))
    if which == 'sirna':
        cons[idx] = 'x'
    else:
        cons[len(seq1) + 1 + idx] = 'x'
    cons_str = ''.join(cons)
    combined = f"{seq1}&{seq2}"
    try:
        proc = subprocess.run(
            ["RNAcofold", "--noPS", "-C"],
            input=f"{combined}\n{cons_str}\n",
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        if not _rnacofold_missing_warned:
            print("WARNING: RNAcofold not found. Duplex energies will be NaN.")
            _rnacofold_missing_warned = True
        return None
    if proc.returncode != 0:
        return None
    return _parse_energy_from_output(proc.stdout)


def calculate_duplex_energy_contributions(seq1: str, seq2: str) -> List[float]:
    """Duplex per-base effects via RNAcofold constraints.

    Returns [total_energy, contrib_sirna_pos1..pos19, contrib_target_pos1..pos19].
    Contribution definition matches single-strand case.
    """
    base = _rnacofold_mfe(seq1, seq2)
    if base is None:
        return [np.nan] + [np.nan] * (len(seq1) + len(seq2))
    out = [float(base)]
    for i in range(len(seq1)):
        ei = _rnacofold_mfe_with_unpaired(seq1, seq2, i, 'sirna')
        out.append(np.nan if ei is None else float(-(ei - base)))
    for j in range(len(seq2)):
        ej = _rnacofold_mfe_with_unpaired(seq1, seq2, j, 'target')
        out.append(np.nan if ej is None else float(-(ej - base)))
    return out


def try_rnaup_energies(sirna_seq: str, target_seq: str) -> Tuple[float, float, float]:
    """Compute RNAup energies (opening, interaction, total) for siRNA-target pair.

    Returns a tuple (opening_dG, interaction_dG, total_dG).
    Raises exception if RNAup is unavailable or parsing fails.
    """
    # Build input: two sequences, each on its own line
    input_data = f"{sirna_seq}\n{target_seq}\n"

    try:
        proc = subprocess.run(
            ["RNAup", "-b", "-d2", "--noLP", "-c", "S"],
            input=input_data,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        global _rnaup_missing_warned
        if not _rnaup_missing_warned:
            print("WARNING: RNAup not found. RNAup features will be NaN.")
            _rnaup_missing_warned = True
        return (np.nan, np.nan, np.nan)

    if proc.returncode != 0:
        raise RuntimeError(f"RNAup failed: {proc.stderr}")

    out = proc.stdout or ''
    # Find last parenthesized energy summary line "( total = open1 + open2 + inter )"
    summary = None
    for l in out.strip().splitlines()[::-1]:
        if '(' in l and ')' in l and '=' in l and '+' in l:
            summary = l[l.rfind('(')+1:l.rfind(')')]
            break
    if not summary:
        raise ValueError("Could not parse RNAup output")
    nums = re.findall(r"[+-]?(?:\d+\.\d+|\d+|\.\d+)", summary)
    vals = [float(x) for x in nums]
    # Expected formats:
    #  - total = open1 + open2 + interaction  (RNAlib >=2.4 typical)
    #  - total = opening + interaction        (older/alt formats)
    if len(vals) >= 4:
        total, open1, open2, interaction = vals[0], vals[1], vals[2], vals[3]
        opening = open1 + open2
    elif len(vals) == 3:
        total, opening, interaction = vals
    else:
        raise ValueError("Unexpected RNAup energy format")
    return (float(opening), float(interaction), float(total))


def build_unified_features(
    df: pd.DataFrame,
    include_rnaup: bool = True,
    include_cofold: bool = True,
    include_advanced_energies: bool = True,
    include_std_seqs: bool = False,
) -> pd.DataFrame:
    """Build comprehensive feature dataset with optimal feature selection.

    Args:
        df: Input DataFrame with siRNA and target columns
        include_rnaup: Include RNAup interaction energies
        include_cofold: Include RNAcofold duplex MFE
        include_advanced_energies: Use constraint-based energy contributions
        include_std_seqs: Include standardized sequences in output

    Returns:
        DataFrame with comprehensive feature set
    """
    sirna_col, target_col = 'siRNA', 'mRNA'
    # Caches are handled internally via lru_cache decorators

    # Prepare outputs
    out_rows = []

    # Keep a curated subset of oligoformer composition features
    f2_names = fc.get_feature_names()
    include_names = {
        "ends", "DH_all",
        "U_all", "G_all", "UU_all", "GG_all", "GC_all", "CC_all", "UA_all",
    }

    # Feature names for new energy calculations
    energy_names = []
    if include_advanced_energies:
        energy_names = ["single_energy_total"] + [f"single_energy_pos{i}" for i in range(1, 20)]
        energy_names += ["duplex_energy_total"] + [f"duplex_energy_sirna_pos{i}" for i in range(1, 20)]
        energy_names += [f"duplex_energy_target_pos{i}" for i in range(1, 20)]

    # RNA interaction energy names
    rnaup_names = ["RNAup_open_dG", "RNAup_interaction_dG"]

    # No embeddings in this version

    print(f"Processing {len(df)} sequences...")
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(df)}")

        sirna_seq = _std_seq(str(row[sirna_col]))
        target_seq = _std_seq(str(row[target_col]))

        # Composition features (oligoformer)
        f2_vals = fc.calculate_oligoformer_features_exact(sirna_seq)
        comp_dict = {name: val for name, val in zip(f2_names, f2_vals) if name in include_names}
        # Round composition features to 2 decimals
        comp_dict = {k: (np.nan if (isinstance(v, float) and np.isnan(v)) else (round(float(v), 2) if isinstance(v, (int, float, np.floating)) else v)) for k, v in comp_dict.items()}

        # Advanced energy features
        energy_dict = {}
        if include_advanced_energies:
            # Single-strand folding energies
            single_energies = calculate_single_energy_contributions(sirna_seq)
            for i, energy in enumerate(single_energies):
                if i == 0:
                    energy_dict["single_energy_total"] = round(energy, 2)
                else:
                    energy_dict[f"single_energy_pos{i}"] = round(energy, 2)

            # Duplex interaction energies
            duplex_energies = calculate_duplex_energy_contributions(sirna_seq, target_seq)
            for i, energy in enumerate(duplex_energies):
                if i == 0:
                    energy_dict["duplex_energy_total"] = round(energy, 2)
                elif i <= 19:  # siRNA positions
                    energy_dict[f"duplex_energy_sirna_pos{i}"] = round(energy, 2)
                else:  # target positions
                    pos = i - 19
                    energy_dict[f"duplex_energy_target_pos{pos}"] = round(energy, 2)

        # RNA tool energies
        rna_tool_dict = {}
        # Skip RNAcofold MFE alias column (duplex_cofold_dG); duplex_energy_total already present

        if include_rnaup:
            try:
                rnaup = try_rnaup_energies(sirna_seq, target_seq)
                rna_tool_dict[rnaup_names[0]] = round(rnaup[0], 2) if not np.isnan(rnaup[0]) else np.nan
                rna_tool_dict[rnaup_names[1]] = round(rnaup[1], 2) if not np.isnan(rnaup[1]) else np.nan
            except Exception:
                rna_tool_dict[rnaup_names[0]] = np.nan
                rna_tool_dict[rnaup_names[1]] = np.nan

        # Sequences
        seq_dict = {}
        if include_std_seqs:
            seq_dict['sirna_seq'] = sirna_seq
            seq_dict['target_seq'] = target_seq

        # Combine all features
        out_rows.append({**seq_dict, **comp_dict, **energy_dict, **rna_tool_dict})

    out_df = pd.DataFrame(out_rows)
    return pd.concat([df.reset_index(drop=True), out_df], axis=1)


def main():
    p = argparse.ArgumentParser(description='Build comprehensive siRNA–target feature dataset with optimal features.')
    p.add_argument('input', nargs='?', default='siRBench_base.csv', help='Input CSV with siRNA and target/mRNA columns (default: siRBench_base.csv)')
    p.add_argument('-o', '--output', help='Output CSV file (default: siRBench_enhanced.csv)')
    p.add_argument('--no-rnaup', action='store_true', help='Disable RNAup energies')
    p.add_argument('--no-cofold', action='store_true', help='Disable RNAcofold MFE calculation')
    p.add_argument('--no-advanced-energies', action='store_true', help='Disable constraint-based energy contributions (RNAfold/RNAcofold)')
    p.add_argument('--with-std-seqs', action='store_true', help='Include standardized siRNA/target sequences in output')
    args = p.parse_args()

    # Set default output filename if not provided
    if not args.output:
        if args.input == 'siRBench_base.csv':
            args.output = 'siRBench_enhanced.csv'
        else:
            base_name = args.input.rsplit('.', 1)[0]
            args.output = f'{base_name}_enhanced.csv'

    input_path = args.input
    if not os.path.isabs(input_path):
        # Look for file in current directory first, then in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(input_path) and os.path.exists(os.path.join(script_dir, '..', input_path)):
            input_path = os.path.join(script_dir, '..', input_path)

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} sequences")
    print(f"Columns: {list(df.columns)}")

    # Show sample of data for siRBench format
    if 'siRNA' in df.columns and 'mRNA' in df.columns:
        print("\nSample sequences:")
        for i in range(min(3, len(df))):
            sirna = df.iloc[i]['siRNA']
            mrna = df.iloc[i]['mRNA']
            efficacy = df.iloc[i].get('efficacy', 'N/A')
            print(f"  {i+1}: siRNA={sirna[:20]}... mRNA={mrna[:20]}... efficacy={efficacy}")

    print("\nBuilding comprehensive feature set...")
    print(f"  - Composition features: YES")
    print(f"  - Advanced energies: {'NO' if args.no_advanced_energies else 'YES (ViennaRNA folding)'}")
    print(f"  - RNAup energies: {'NO' if args.no_rnaup else 'YES'}")
    print(f"  - RNAcofold MFE column: NO (use duplex_energy_total)")
    # Embeddings removed

    out = build_unified_features(
        df,
        include_rnaup=not args.no_rnaup,
        include_cofold=not args.no_cofold,
        include_advanced_energies=not args.no_advanced_energies,
        include_std_seqs=args.with_std_seqs,
    )

    output_path = args.output
    if not os.path.isabs(output_path):
        # Save to script's parent directory (data folder) if relative path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, '..', output_path)

    print(f"\nSaving {out.shape[1]} features to {output_path}...")
    # Ensure consistent 2-decimal formatting for all float columns
    out.to_csv(output_path, index=False, float_format='%.2f')
    print(f"Enhanced dataset saved with {out.shape[0]} sequences and {out.shape[1]} features")
    print("Done!")


if __name__ == '__main__':
    main()
