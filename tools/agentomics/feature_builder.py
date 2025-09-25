import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_cols_ = None
        self.feature_names_ = None
        self.group_columns_ = {
            "DG_pos": [f"DG_pos{i}" for i in range(1, 19)],
            "DH_pos": [f"DH_pos{i}" for i in range(1, 19)],
            "single_energy_pos": [f"single_energy_pos{i}" for i in range(1, 20)],
            "duplex_energy_sirna_pos": [f"duplex_energy_sirna_pos{i}" for i in range(1, 20)],
            "duplex_energy_target_pos": [f"duplex_energy_target_pos{i}" for i in range(1, 20)],
        }

    def fit(self, X, y=None):
        df = X.copy()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        self.numeric_cols_ = [c for c in numeric_cols if c not in {"target", "numeric_label"}]
        features = self._build_features(df)
        self.feature_names_ = features.columns.tolist()
        return self

    def transform(self, X):
        df = X.copy()
        features = self._build_features(df)
        return features[self.feature_names_]

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        base = df[self.numeric_cols_].copy()
        self._add_group_stats(base)
        self._add_interaction_features(base)
        base.fillna(0.0, inplace=True)
        return base

    def _add_group_stats(self, base: pd.DataFrame) -> None:
        for group_name, columns in self.group_columns_.items():
            available = [c for c in columns if c in base.columns]
            if len(available) < 2:
                continue
            subset = base[available]
            prefix = f"{group_name}"
            base[f"{prefix}_mean"] = subset.mean(axis=1)
            base[f"{prefix}_std"] = subset.std(axis=1, ddof=0)
            base[f"{prefix}_min"] = subset.min(axis=1)
            base[f"{prefix}_max"] = subset.max(axis=1)
            segments = [seg.tolist() for seg in np.array_split(available, 3) if len(seg) > 0]
            segment_names = ["seed", "mid", "tail"]
            active_segment_names = []
            for name, seg_cols in zip(segment_names, segments):
                seg_subset = subset[seg_cols]
                base[f"{prefix}_mean_{name}"] = seg_subset.mean(axis=1)
                base[f"{prefix}_std_{name}"] = seg_subset.std(axis=1, ddof=0)
                active_segment_names.append(name)
            if "seed" in active_segment_names and "tail" in active_segment_names:
                base[f"{prefix}_seed_tail_mean_diff"] = base[f"{prefix}_mean_seed"] - base[f"{prefix}_mean_tail"]
            first_col, last_col = available[0], available[-1]
            base[f"{prefix}_first_last_diff"] = base[first_col] - base[last_col]
            weights = np.linspace(0.0, 1.0, num=len(available))
            weighted = (subset.values * weights).sum(axis=1)
            base[f"{prefix}_weighted_trend"] = weighted

    def _add_interaction_features(self, base: pd.DataFrame) -> None:
        if {"single_energy_total", "duplex_energy_total"}.issubset(base.columns):
            base["single_duplex_gap"] = base["single_energy_total"] - base["duplex_energy_total"]
        if {"RNAup_open_dG", "RNAup_interaction_dG"}.issubset(base.columns):
            base["RNAup_combined_dG"] = base["RNAup_open_dG"] + base["RNAup_interaction_dG"]
        if {"DG_total", "DH_total"}.issubset(base.columns):
            base["DG_DH_diff"] = base["DG_total"] - base["DH_total"]
        dg_mean = base.get("DG_pos_mean")
        dh_mean = base.get("DH_pos_mean")
        if dg_mean is not None and dh_mean is not None:
            base["DG_DH_mean_diff"] = dg_mean - dh_mean
        sirna_mean = base.get("duplex_energy_sirna_pos_mean")
        target_mean = base.get("duplex_energy_target_pos_mean")
        if sirna_mean is not None and target_mean is not None:
            base["duplex_sirna_target_mean_gap"] = sirna_mean - target_mean
