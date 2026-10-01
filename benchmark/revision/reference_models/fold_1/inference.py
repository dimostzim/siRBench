"""Portable CPU inference for the selected single RBF SVR."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '16'
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from augmented_representation import AugmentedNormalizer

DEFAULT_ARTIFACTS = Path(__file__).resolve().parent / 'training_artifacts'

def predict(df, artifacts=DEFAULT_ARTIFACTS):
    required = {'id', 'siRNA', 'extended_mRNA'}
    if not required.issubset(df.columns):
        raise ValueError('Input requires id, siRNA, extended_mRNA columns')
    artifacts = Path(artifacts)
    normalizer = AugmentedNormalizer.load(artifacts)
    model = joblib.load(artifacts / 'model.joblib')
    predictions = np.empty(len(df), dtype=np.float64)
    for start in range(0, len(df), 256):
        batch = df.iloc[start:start + 256]
        predictions[start:start + len(batch)] = model.predict(normalizer.encode_transform(batch))
    if not np.isfinite(predictions).all():
        raise RuntimeError('Model returned nonfinite predictions')
    return predictions

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--artifacts-dir', default=str(DEFAULT_ARTIFACTS))
    args = parser.parse_args()
    df = pd.read_csv(Path(args.input) / 'data.csv', dtype=str, keep_default_na=False)
    predictions = predict(df, args.artifacts_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id': df['id'].to_numpy(), 'prediction': predictions}).to_csv(output, index=False)

if __name__ == '__main__':
    main()
