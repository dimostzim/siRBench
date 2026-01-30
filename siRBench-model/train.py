import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from feature_builder import build_feature_matrix


def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pearson = float(stats.pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else float('nan')
    spearman = float(stats.spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else float('nan')
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson,
        'spearman': spearman,
    }


def quintile_bias(y_true, y_pred, q=5):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df['quintile'] = pd.qcut(df['y_true'], q, labels=False, duplicates='drop')
    rows = []
    for g, sub in df.groupby('quintile'):
        t_mean = sub['y_true'].mean()
        p_mean = sub['y_pred'].mean()
        bias = p_mean - t_mean
        rmse = np.sqrt(mean_squared_error(sub['y_true'], sub['y_pred']))
        rows.append({'quintile': int(g), 'true_mean': float(t_mean), 'pred_mean': float(p_mean), 'bias': float(bias), 'rmse': float(rmse), 'count': len(sub)})
    return rows


def group_bias(df, group_col):
    rows = []
    for g, sub in df.groupby(group_col):
        t_mean = sub['y_true'].mean()
        p_mean = sub['y_pred'].mean()
        bias = p_mean - t_mean
        rows.append({group_col: g, 'true_mean': float(t_mean), 'pred_mean': float(p_mean), 'bias': float(bias), 'count': len(sub)})
    return rows


def try_xgb_gpu():
    # Attempt to use GPU; fallback handled by caller if training fails
    return 'gpu_hist', 'gpu_predictor'


def train(train_path: str, val_path: str, artifacts_dir: str):
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Build features
    X_train, feature_names, encoder = build_feature_matrix(train_df, fit_encoder=True, artifacts_path=os.path.join(artifacts_dir, 'feature_artifacts.json'))
    X_val, _, _ = build_feature_matrix(val_df, encoder=encoder, fit_encoder=False)

    # Use efficacy column for training targets
    y_train = train_df['efficacy'].to_numpy(dtype=np.float32)
    y_val = val_df['efficacy'].to_numpy(dtype=np.float32)

    sample_weight_train = 1.0 + 3.5 * np.abs(y_train - 0.5)
    sample_weight_val = 1.0 + 3.5 * np.abs(y_val - 0.5)

    # XGBoost DART
    tree_method, predictor = try_xgb_gpu()
    xgb_params = dict(
        n_estimators=2600,
        learning_rate=0.028,
        max_depth=7,
        min_child_weight=1.2,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        booster='dart',
        sample_type='uniform',
        normalize_type='tree',
        rate_drop=0.1,
        skip_drop=0.5,
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method=tree_method,
        predictor=predictor,
        n_jobs=-1,
        random_state=42,
    )
    try:
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[sample_weight_val],
            verbose=False,
            early_stopping_rounds=250,
        )
    except Exception:
        # Fallback to CPU
        xgb_params['tree_method'] = 'hist'
        xgb_params['predictor'] = 'auto'
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[sample_weight_val],
            verbose=False,
            early_stopping_rounds=250,
        )
    xgb_model.save_model(os.path.join(artifacts_dir, 'xgb_model.json'))

    # LightGBM
    lgb_device = 'gpu'
    lgb_params = dict(
        objective='regression',
        metric='rmse',
        boosting_type='gbdt',
        device=lgb_device,
        n_estimators=7000,
        learning_rate=0.01,
        max_depth=9,
        num_leaves=640,
        min_child_samples=10,
        min_child_weight=1e-3,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=0.7,
        random_state=42,
        n_jobs=-1,
    )
    lgb_callbacks = [lgb.early_stopping(500, verbose=False)]
    try:
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[sample_weight_val],
            eval_metric='rmse',
            callbacks=lgb_callbacks,
        )
    except Exception:
        lgb_params['device'] = 'cpu'
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[sample_weight_val],
            eval_metric='rmse',
            callbacks=lgb_callbacks,
        )
    booster = lgb_model.booster_
    booster.save_model(os.path.join(artifacts_dir, 'lgbm_model.txt'), num_iteration=lgb_model.best_iteration_)

    # Predictions
    xgb_pred = xgb_model.predict(X_val)
    lgb_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)
    avg_pred = (xgb_pred + lgb_pred) / 2.0

    # Calibration
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(avg_pred, y_val)
    calibrated_pred = calibrator.predict(avg_pred)
    calibrated_pred = np.clip(calibrated_pred, 0.0, 1.0)
    joblib.dump(calibrator, os.path.join(artifacts_dir, 'calibrator.joblib'))

    # Metrics
    metrics = compute_metrics(y_val, calibrated_pred)
    with open(os.path.join(artifacts_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Quintile bias
    quint_bias = quintile_bias(y_val, calibrated_pred)
    pd.DataFrame(quint_bias).to_csv(os.path.join(artifacts_dir, 'quintile_bias.csv'), index=False)

    # Group bias for cell_line and source
    bias_tables = {}
    val_out = val_df.copy()
    val_out['y_true'] = y_val
    val_out['y_pred'] = calibrated_pred
    for col in ['cell_line', 'source']:
        bias = group_bias(val_out, col)
        bias_tables[col] = bias
        pd.DataFrame(bias).to_csv(os.path.join(artifacts_dir, f'{col}_bias.csv'), index=False)

    # Save predictions
    preds_df = pd.DataFrame({
        'id': val_df['id'] if 'id' in val_df.columns else np.arange(len(y_val)),
        'y_true': y_val,
        'xgb_pred': xgb_pred,
        'lgb_pred': lgb_pred,
        'avg_pred': avg_pred,
        'calibrated_pred': calibrated_pred,
    })
    preds_df.to_csv(os.path.join(artifacts_dir, 'validation_predictions.csv'), index=False)

    # Save summary
    summary = {
        'metrics': metrics,
        'quintile_bias': quint_bias,
        'cell_line_bias': bias_tables.get('cell_line', []),
        'source_bias': bias_tables.get('source', []),
        'xgb_best_iteration': int(getattr(xgb_model, 'best_iteration', None) or xgb_model.get_booster().best_ntree_limit),
        'lgb_best_iteration': int(lgb_model.best_iteration_ if lgb_model.best_iteration_ is not None else lgb_model.n_estimators),
        'feature_count': int(X_train.shape[1]),
    }
    with open(os.path.join(artifacts_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--validation-data', required=True)
    parser.add_argument('--artifacts-dir', required=True)
    args = parser.parse_args()

    train(args.train_data, args.validation_data, args.artifacts_dir)


if __name__ == '__main__':
    main()
