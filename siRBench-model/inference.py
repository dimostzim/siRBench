import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder

# Reuse feature builder utilities
from feature_builder import build_feature_matrix, BASES


def load_artifacts(artifacts_dir: Path):
    artifacts_dir = Path(artifacts_dir)
    feat_art_path = artifacts_dir / 'feature_artifacts.json'
    with open(feat_art_path, 'r') as f:
        feat_artifacts = json.load(f)
    categories = feat_artifacts['categories']
    numeric_cols = feat_artifacts['numeric_cols']
    feature_names = feat_artifacts.get('feature_names', None)

    # Rebuild encoder using stored categories, fitting on a tiny dummy dataset to populate
    # sklearn internal attributes for compatibility across versions.
    encoder = OneHotEncoder(categories=[np.array(c) for c in categories], handle_unknown='ignore', sparse_output=False)
    dummy = pd.DataFrame({
        'source': [categories[0][0] if len(categories[0]) > 0 else ''],
        'cell_line': [categories[1][0] if len(categories[1]) > 0 else ''],
    })
    encoder.fit(dummy)

    # Load models
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(artifacts_dir / 'xgb_model.json')
    # Try to prefer GPU predictor if available
    try:
        xgb_model.set_params(predictor='gpu_predictor')
    except Exception:
        pass

    lgb_model = lgb.Booster(model_file=str(artifacts_dir / 'lgbm_model.txt'))

    calibrator = joblib.load(artifacts_dir / 'calibrator.joblib')

    return encoder, numeric_cols, feature_names, xgb_model, lgb_model, calibrator


def ensure_numeric_cols(df: pd.DataFrame, numeric_cols):
    df = df.copy()
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df


def prepare_dataframe(df: pd.DataFrame, numeric_cols):
    df = df.copy()
    # Ensure required columns
    if 'siRNA' not in df.columns or 'mRNA' not in df.columns:
        raise ValueError('Input must contain siRNA and mRNA columns')
    if 'extended_mRNA' not in df.columns:
        df['extended_mRNA'] = 'N'
    if 'source' not in df.columns:
        df['source'] = 'unknown'
    if 'cell_line' not in df.columns:
        df['cell_line'] = 'unknown'

    df = ensure_numeric_cols(df, numeric_cols)

    # Keep only the columns used by the feature builder to avoid drift
    keep_cols = ['siRNA', 'mRNA', 'extended_mRNA', 'source', 'cell_line', 'id'] + list(numeric_cols)
    # Filter
    existing_keep = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep]

    return df


def build_features(df: pd.DataFrame, encoder: OneHotEncoder, numeric_cols):
    df_prepared = prepare_dataframe(df, numeric_cols)
    X, feature_names, _ = build_feature_matrix(df_prepared, encoder=encoder, fit_encoder=False)
    return X, feature_names


def predict(df: pd.DataFrame, encoder, numeric_cols, xgb_model, lgb_model, calibrator):
    X, _ = build_features(df, encoder, numeric_cols)
    # XGBoost prediction
    try:
        xgb_pred = xgb_model.predict(X, output_margin=False)
    except Exception:
        # fallback by reloading booster and using DMatrix
        booster = xgb.Booster()
        booster.load_model(xgb_model.get_booster().save_raw())
        dmat = xgb.DMatrix(X)
        xgb_pred = booster.predict(dmat)
    # LightGBM prediction
    try:
        lgb_pred = lgb_model.predict(X)
    except Exception:
        # If Booster not compatible, load as sklearn API
        lgb_reg = lgb.LGBMRegressor()
        lgb_reg._Booster = lgb_model
        lgb_pred = lgb_reg.predict(X)

    avg_pred = (np.asarray(xgb_pred) + np.asarray(lgb_pred)) / 2.0
    calibrated = calibrator.predict(avg_pred)
    calibrated = np.clip(calibrated, 0.0, 1.0)
    return calibrated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input CSV path')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--artifacts-dir', default='./training_artifacts', help='Artifacts directory')
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    encoder, numeric_cols, feature_names, xgb_model, lgb_model, calibrator = load_artifacts(artifacts_dir)

    df = pd.read_csv(args.input)
    # Require siRNA and mRNA columns
    missing = [c for c in ['siRNA', 'mRNA'] if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    preds = predict(df, encoder, numeric_cols, xgb_model, lgb_model, calibrator)

    out_df = pd.DataFrame({'prediction': preds})
    if 'id' in df.columns:
        out_df.insert(0, 'id', df['id'])
    out_df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
