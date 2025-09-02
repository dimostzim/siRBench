#!/usr/bin/env python3

import pandas as pd
import argparse
from features_calculator import calculate_oligoformer_features_exact, calculate_duplex_folding_energy, get_feature_names

def process_sirna_data(input_file, output_file):
    df = pd.read_csv(input_file)
    
    sirna_col = None
    target_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'sirna' in col_lower:
            sirna_col = col
        elif 'mrna' in col_lower or 'target' in col_lower:
            target_col = col
    
    feature_names = get_feature_names()
    feature_data = []
    duplex_energies = []
    
    for idx, row in df.iterrows():
        sirna_seq = str(row[sirna_col]).upper().replace('T', 'U')
        target_seq = str(row[target_col]).upper().replace('T', 'U')
        
        features = calculate_oligoformer_features_exact(sirna_seq)
        features = [round(f, 2) if f is not None else None for f in features]
        feature_data.append(features)
        
        duplex_energy = calculate_duplex_folding_energy(sirna_seq, target_seq)
        duplex_energies.append(round(duplex_energy, 2) if duplex_energy is not None else None)
    
    for i, feature_name in enumerate(feature_names):
        df[feature_name] = [row[i] for row in feature_data]
    
    df['duplex_folding_dG'] = duplex_energies
    df.to_csv(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description='Process siRNA data with thermodynamic features')
    parser.add_argument('input_file', help='Input CSV file with siRNA and target sequences')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    process_sirna_data(args.input_file, args.output)

if __name__ == "__main__":
    main()