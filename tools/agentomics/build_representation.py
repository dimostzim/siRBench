from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"

DEFAULT_TRAIN = DATA_DIR / "train.csv"
DEFAULT_VAL = DATA_DIR / "validation.csv"
SIRBENCH_TRAIN = DATA_DIR / "siRBench_train_90.csv"
SIRBENCH_VAL = DATA_DIR / "siRBench_val_10.csv"

TRAIN_PATH = DEFAULT_TRAIN if DEFAULT_TRAIN.exists() else SIRBENCH_TRAIN
VAL_PATH = DEFAULT_VAL if DEFAULT_VAL.exists() else SIRBENCH_VAL
OUTPUT_PATH = BASE_DIR / "representation.joblib"

SEQ_COLS = ["siRNA", "mRNA", "extended_mRNA"]
TARGET_CANDIDATES = ["efficacy", "target"]
LEAK_CANDIDATES = ["binary", "source", "cell_line", "numeric_label"]

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

TARGET_COL = next((col for col in TARGET_CANDIDATES if col in train_df.columns), None)
if TARGET_COL is None:
    raise KeyError(f"No target column found in {TRAIN_PATH}; expected one of {TARGET_CANDIDATES}")

drop_cols = SEQ_COLS
drop_cols += [col for col in TARGET_CANDIDATES if col in train_df.columns]
drop_cols += [col for col in LEAK_CANDIDATES if col in train_df.columns]
base_numeric_cols = [c for c in train_df.columns if c not in drop_cols]
base_numeric_cols.sort()


def add_positional_stats(df: pd.DataFrame, prefix: str, max_pos: int, added: list):
    cols = [f"{prefix}{i}" for i in range(1, max_pos + 1)]
    if not set(cols).issubset(df.columns):
        return
    values = df[cols].to_numpy(dtype=float)
    df[f"{prefix}mean_all"] = values.mean(axis=1)
    df[f"{prefix}std_all"] = values.std(axis=1)
    added.extend([f"{prefix}mean_all", f"{prefix}std_all"])

    segments = {
        "seed": [i for i in range(1, 8) if i <= max_pos],
        "mid": [i for i in range(8, 15) if i <= max_pos],
        "tail": [i for i in range(15, max_pos + 1) if i <= max_pos],
    }
    for seg_name, seg_positions in segments.items():
        if not seg_positions:
            continue
        seg_cols = [f"{prefix}{i}" for i in seg_positions]
        df[f"{prefix}mean_{seg_name}"] = df[seg_cols].mean(axis=1)
        added.append(f"{prefix}mean_{seg_name}")
    seed_col = f"{prefix}mean_seed"
    tail_col = f"{prefix}mean_tail"
    if seed_col in df.columns and tail_col in df.columns:
        diff_name = f"{prefix}seed_tail_diff"
        df[diff_name] = df[seed_col] - df[tail_col]
        added.append(diff_name)


def build_feature_frame(df: pd.DataFrame):
    feats = df[base_numeric_cols].copy()
    added_features = []

    add_positional_stats(feats, "DG_pos", 18, added_features)
    add_positional_stats(feats, "DH_pos", 18, added_features)
    add_positional_stats(feats, "single_energy_pos", 19, added_features)
    add_positional_stats(feats, "duplex_energy_sirna_pos", 19, added_features)
    add_positional_stats(feats, "duplex_energy_target_pos", 19, added_features)

    if {"single_energy_total", "duplex_energy_total"}.issubset(feats.columns):
        feats["single_duplex_gap"] = feats["single_energy_total"] - feats["duplex_energy_total"]
        added_features.append("single_duplex_gap")
    if {"RNAup_open_dG", "RNAup_interaction_dG"}.issubset(feats.columns):
        feats["RNAup_combined_dG"] = feats["RNAup_open_dG"] + feats["RNAup_interaction_dG"]
        added_features.append("RNAup_combined_dG")

    return feats, added_features


train_features, added_train_features = build_feature_frame(train_df)
val_features, _ = build_feature_frame(val_df)

feature_names = train_features.columns.tolist()
val_features = val_features[feature_names]

scaler = StandardScaler()
X_train = scaler.fit_transform(train_features).astype(np.float32)
X_val = scaler.transform(val_features).astype(np.float32)
y_train = train_df[TARGET_COL].to_numpy(dtype=np.float32)
y_val = val_df[TARGET_COL].to_numpy(dtype=np.float32)

representation = {
    "feature_names": feature_names,
    "base_numeric_cols": base_numeric_cols,
    "added_features": added_train_features,
    "scaler": scaler,
    "X_train": X_train,
    "X_val": X_val,
    "y_train": y_train,
    "y_val": y_val,
}

joblib.dump(representation, OUTPUT_PATH)

print(f"base_features={len(base_numeric_cols)} added_features={len(added_train_features)} total_features={len(feature_names)}")
