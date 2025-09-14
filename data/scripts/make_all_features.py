#!/usr/bin/env python3

"""
Build a comprehensive siRNA–target feature dataset with optimal feature selection.

Combines the best features from multiple approaches:
- Sequence composition features (oligoformer): U_all, G_all, dinucleotide counts, etc.
- Advanced energy calculations: Per-base folding contributions via ViennaRNA
- RNA foundation model embeddings: 640-dim learned representations
- RNA interaction energies: RNAup and RNAcofold calculations

Replaces simple ΔG lookup with sophisticated folding-based energy contributions.
Adds optional RNA-FM transformer embeddings for enhanced prediction capability.

Inputs: CSV with columns containing "sirna" and "mrna" or "target" in the names.
Outputs: CSV with comprehensive feature set optimized for siRNA efficacy prediction.
"""

import argparse
import os
import sys
import subprocess
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np

# RNA analysis imports - fail early if not available
import RNA

# Optional imports for embeddings
try:
    import torch
    import fm
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

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


def setup_rna_fm():
    """Initialize RNA foundation model components."""
    if not EMBEDDINGS_AVAILABLE:
        raise ImportError("RNA-FM dependencies (torch, fm) not available")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device=device)
    return model, batch_converter, device


def calculate_single_energy_contributions(sequence: str) -> List[float]:
    """Calculate per-base energy contributions using ViennaRNA folding.

    Returns list where first element is total MFE, followed by
    energy contribution of each base (calculated by mutation analysis).
    """
    sequence = sequence.replace('\n', '').replace('T', 'U')

    # Calculate total MFE
    fc = RNA.fold_compound(sequence)
    (ss, mfe) = fc.mfe()
    contributions = [float(mfe)]

    # Calculate per-base contributions
    for i in range(len(sequence)):
        # Mutate base to 'X' and recalculate
        mutated_seq = sequence[:i] + 'X' + sequence[i+1:]
        fc_mut = RNA.fold_compound(mutated_seq)
        (_, mfe_mut) = fc_mut.mfe()

        # Contribution = change in stability when base is removed
        contribution = mfe_mut - mfe
        contributions.append(float(-contribution))  # Negative for intuitive interpretation

    return contributions


def calculate_duplex_energy_contributions(seq1: str, seq2: str) -> List[float]:
    """Calculate duplex energy contributions using ViennaRNA duplexfold.

    Returns [total_energy, seq1_base_contributions..., seq2_base_contributions...]
    """
    def _duplex_energy(s1: str, s2: str) -> float:
        duplexes = RNA.duplexfold(s1, s2)
        return float(duplexes.energy)

    # Total duplex energy
    total_energy = _duplex_energy(seq1, seq2)
    contributions = [total_energy]

    # Seq1 base contributions
    for i in range(len(seq1)):
        mutated_seq1 = seq1[:i] + 'X' + seq1[i+1:]
        mutated_energy = _duplex_energy(mutated_seq1, seq2)
        contribution = mutated_energy - total_energy
        contributions.append(float(-contribution))

    # Seq2 base contributions
    for i in range(len(seq2)):
        mutated_seq2 = seq2[:i] + 'X' + seq2[i+1:]
        mutated_energy = _duplex_energy(seq1, mutated_seq2)
        contribution = mutated_energy - total_energy
        contributions.append(float(-contribution))

    return contributions


def calculate_rna_fm_embeddings(sequences: List[Tuple[str, str]], rna_fm_components) -> np.ndarray:
    """Calculate RNA-FM embeddings for sequences.

    Args:
        sequences: List of (name, sequence) tuples
        rna_fm_components: (model, batch_converter, device) tuple

    Returns:
        Embedding array of shape (sequence_length, 640)
    """
    if not EMBEDDINGS_AVAILABLE:
        raise ImportError("RNA-FM dependencies not available")

    model, batch_converter, device = rna_fm_components

    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)

    with torch.no_grad():
        results = model(batch_tokens.to(device=device), repr_layers=[12])
        embeddings = results["representations"][12][0][:-1].cpu().numpy()  # Remove EOS token

    return embeddings


def try_rnaup_energies(sirna_seq: str, target_seq: str) -> Tuple[float, float, float]:
    """Compute RNAup energies (opening, interaction, total) for siRNA-target pair.

    Returns a tuple (opening_dG, interaction_dG, total_dG).
    Raises exception if RNAup is unavailable or parsing fails.
    """
    # Build input: two sequences, each on its own line
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
    # Heuristic parse: look for the last parenthesized triple; extract numbers
    # Example patterns often include lines like:  "... ( -X = -Y + -Z )"
    line = None
    for l in out.strip().splitlines()[::-1]:
        if '(' in l and ')' in l and '=' in l:
            line = l
            break

    if not line:
        raise ValueError("Could not parse RNAup output")

    # Extract inside parentheses and split numbers
    start = line.rfind('(')
    end = line.rfind(')')
    inner = line[start + 1:end]
    # Remove symbols and split
    cleaned = inner.replace('=', ' ').replace('+', ' ')
    parts = [p for p in cleaned.split() if p]
    # Usually order is: total, opening, interaction; reorder to (opening, interaction, total)
    vals = [float(x) for x in parts]
    if len(vals) < 3:
        raise ValueError("Insufficient values in RNAup output")

    total = vals[0]
    opening = vals[1]
    interaction = vals[2]
    return float(opening), float(interaction), float(total)


def build_unified_features(
    df: pd.DataFrame,
    include_rnaup: bool = True,
    include_cofold: bool = True,
    include_advanced_energies: bool = True,
    include_embeddings: bool = False,
    include_std_seqs: bool = False,
) -> pd.DataFrame:
    """Build comprehensive feature dataset with optimal feature selection.

    Args:
        df: Input DataFrame with siRNA and target columns
        include_rnaup: Include RNAup interaction energies
        include_cofold: Include RNAcofold duplex MFE
        include_advanced_energies: Use ViennaRNA folding-based energy contributions
        include_embeddings: Include RNA-FM transformer embeddings (requires GPU/time)
        include_std_seqs: Include standardized sequences in output

    Returns:
        DataFrame with comprehensive feature set
    """
    sirna_col, target_col = 'siRNA', 'mRNA'
    rna_fm_components = None
    if include_embeddings:
        if not EMBEDDINGS_AVAILABLE:
            print("WARNING: RNA-FM dependencies not available. Skipping embeddings.")
            include_embeddings = False
        else:
            rna_fm_components = setup_rna_fm()

    # Prepare outputs
    out_rows = []

    # Keep composition features from oligoformer, drop redundant DG features
    f2_names = fc.get_feature_names()
    exclude_names = {"DG_1", "DG_2", "DG_13", "DG_18"}  # Replaced by advanced energies

    # Feature names for new energy calculations
    energy_names = []
    if include_advanced_energies:
        energy_names = ["single_energy_total"] + [f"single_energy_pos{i}" for i in range(1, 20)]
        energy_names += ["duplex_energy_total"] + [f"duplex_energy_sirna_pos{i}" for i in range(1, 20)]
        energy_names += [f"duplex_energy_target_pos{i}" for i in range(1, 20)]

    # RNA interaction energy names
    rnaup_names = ["RNAup_open_dG", "RNAup_interaction_dG", "RNAup_total_dG"]
    cofold_name = "duplex_cofold_dG"

    # Embedding feature names
    embedding_names = []
    if include_embeddings:
        embedding_names = ([f"single_emb_{i}" for i in range(640)] +
                          [f"duplex_emb_{i}" for i in range(640)])

    print(f"Processing {len(df)} sequences...")
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(df)}")

        sirna_seq = _std_seq(str(row[sirna_col]))
        target_seq = _std_seq(str(row[target_col]))

        # Composition features (oligoformer)
        f2_vals = fc.calculate_oligoformer_features_exact(sirna_seq)
        comp_dict = {name: val for name, val in zip(f2_names, f2_vals) if name not in exclude_names}

        # Advanced energy features
        energy_dict = {}
        if include_advanced_energies:
            # Single-strand folding energies
            single_energies = calculate_single_energy_contributions(sirna_seq)
            for i, energy in enumerate(single_energies):
                if i == 0:
                    energy_dict["single_energy_total"] = round(energy, 3)
                else:
                    energy_dict[f"single_energy_pos{i}"] = round(energy, 3)

            # Duplex interaction energies
            duplex_energies = calculate_duplex_energy_contributions(sirna_seq, target_seq)
            for i, energy in enumerate(duplex_energies):
                if i == 0:
                    energy_dict["duplex_energy_total"] = round(energy, 3)
                elif i <= 19:  # siRNA positions
                    energy_dict[f"duplex_energy_sirna_pos{i}"] = round(energy, 3)
                else:  # target positions
                    pos = i - 19
                    energy_dict[f"duplex_energy_target_pos{pos}"] = round(energy, 3)

        # RNA tool energies
        rna_tool_dict = {}
        if include_cofold:
            mfe = fc.calculate_duplex_folding_energy(sirna_seq, target_seq)
            if mfe is None:
                raise RuntimeError("RNAcofold calculation failed")
            rna_tool_dict[cofold_name] = round(mfe, 2)

        if include_rnaup:
            rnaup = try_rnaup_energies(sirna_seq, target_seq)
            rna_tool_dict[rnaup_names[0]] = round(rnaup[0], 2)
            rna_tool_dict[rnaup_names[1]] = round(rnaup[1], 2)
            rna_tool_dict[rnaup_names[2]] = round(rnaup[2], 2)

        # RNA-FM embeddings
        emb_dict = {}
        if include_embeddings and rna_fm_components is not None:
            # Single sequence embeddings
            single_seqs = [("sirna", sirna_seq)]
            single_emb = calculate_rna_fm_embeddings(single_seqs, rna_fm_components)
            for i, val in enumerate(single_emb.flatten()[:640]):
                emb_dict[f"single_emb_{i}"] = float(val)

            # Duplex embeddings
            duplex_seqs = [("duplex", sirna_seq + target_seq)]
            duplex_emb = calculate_rna_fm_embeddings(duplex_seqs, rna_fm_components)
            for i, val in enumerate(duplex_emb.flatten()[:640]):
                emb_dict[f"duplex_emb_{i}"] = float(val)

        # Sequences
        seq_dict = {}
        if include_std_seqs:
            seq_dict['sirna_seq'] = sirna_seq
            seq_dict['target_seq'] = target_seq

        # Combine all features
        out_rows.append({**seq_dict, **comp_dict, **energy_dict, **rna_tool_dict, **emb_dict})

    out_df = pd.DataFrame(out_rows)
    return pd.concat([df.reset_index(drop=True), out_df], axis=1)


def main():
    p = argparse.ArgumentParser(description='Build comprehensive siRNA–target feature dataset with optimal features.')
    p.add_argument('input', nargs='?', default='siRBench_base.csv', help='Input CSV with siRNA and target/mRNA columns (default: siRBench_base.csv)')
    p.add_argument('-o', '--output', help='Output CSV file (default: siRBench_enhanced.csv)')
    p.add_argument('--no-rnaup', action='store_true', help='Disable RNAup energies')
    p.add_argument('--no-cofold', action='store_true', help='Disable RNAcofold MFE calculation')
    p.add_argument('--no-advanced-energies', action='store_true', help='Disable ViennaRNA folding-based energy contributions')
    p.add_argument('--with-embeddings', action='store_true', help='Include RNA-FM transformer embeddings (slower, requires GPU)')
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
    print(f"  - RNAcofold MFE: {'NO' if args.no_cofold else 'YES'}")
    print(f"  - RNA-FM embeddings: {'YES (1280 features)' if args.with_embeddings else 'NO'}")

    out = build_unified_features(
        df,
        include_rnaup=not args.no_rnaup,
        include_cofold=not args.no_cofold,
        include_advanced_energies=not args.no_advanced_energies,
        include_embeddings=args.with_embeddings,
        include_std_seqs=args.with_std_seqs,
    )

    output_path = args.output
    if not os.path.isabs(output_path):
        # Save to script's parent directory (data folder) if relative path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, '..', output_path)

    print(f"\nSaving {out.shape[1]} features to {output_path}...")
    out.to_csv(output_path, index=False)
    print(f"Enhanced dataset saved with {out.shape[0]} sequences and {out.shape[1]} features")
    print("Done!")


if __name__ == '__main__':
    main()