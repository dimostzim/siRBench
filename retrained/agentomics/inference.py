import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
ARTIFACT_PATH = BASE_DIR / "representation.joblib"
MODEL_PATH = BASE_DIR / "model.txt"

BASE_MAP = {"A": 0, "C": 1, "G": 2, "U": 3, "X": 4}
PURINES = {"A", "G"}
PYRIMIDINES = {"C", "U"}
CANONICAL_PAIRS = {"AU", "UA", "GC", "CG", "GU", "UG"}


def _normalize_sequence(seq: str) -> str:
    return (seq or "").upper().replace("T", "U")


def _gc_content(seq: str) -> float:
    seq = _normalize_sequence(seq)
    if not seq:
        return 0.0
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq)


def _base_code(base: str) -> int:
    base = _normalize_sequence(base)
    if not base:
        return -1
    return BASE_MAP.get(base[0], -1)


def _base_fraction_features(seq: str, prefix: str) -> Dict[str, float]:
    seq = _normalize_sequence(seq)
    length = len(seq)
    counts = Counter(seq)
    features: Dict[str, float] = {f"{prefix}_len": float(length)}
    if length == 0:
        for base in ["A", "C", "G", "U", "X"]:
            features[f"{prefix}_frac_{base}"] = 0.0
        features[f"{prefix}_purine_frac"] = 0.0
        features[f"{prefix}_pyrimidine_frac"] = 0.0
        return features
    for base in ["A", "C", "G", "U", "X"]:
        features[f"{prefix}_frac_{base}"] = counts.get(base, 0) / length
    purines = sum(counts.get(b, 0) for b in PURINES)
    pyrimidines = sum(counts.get(b, 0) for b in PYRIMIDINES)
    features[f"{prefix}_purine_frac"] = purines / length
    features[f"{prefix}_pyrimidine_frac"] = pyrimidines / length
    return features


def _region(seq: str, start: int, end: int) -> str:
    seq = _normalize_sequence(seq)
    return seq[start:end]


def _pair_features(sirna: str, mrna: str) -> Dict[str, float]:
    sirna = _normalize_sequence(sirna)
    mrna = _normalize_sequence(mrna)
    length = min(len(sirna), len(mrna))
    if length == 0:
        features = {f"pair_frac_{pair}": 0.0 for pair in CANONICAL_PAIRS}
        features["pair_frac_mismatch"] = 0.0
        features["pair_match_ratio"] = 0.0
        return features
    pair_counts = Counter()
    mismatch = 0
    for s_base, m_base in zip(sirna[:length], mrna[:length]):
        pair = f"{s_base}{m_base}"
        if pair in CANONICAL_PAIRS:
            pair_counts[pair] += 1
        else:
            mismatch += 1
    features = {f"pair_frac_{pair}": pair_counts.get(pair, 0) / length for pair in CANONICAL_PAIRS}
    features["pair_frac_mismatch"] = mismatch / length
    features["pair_match_ratio"] = 1.0 - features["pair_frac_mismatch"]
    return features


def _seed_tail_features(seq: str, prefix: str) -> Dict[str, float]:
    seq = _normalize_sequence(seq)
    seed = _region(seq, 0, 7)
    tail = _region(seq, max(len(seq) - 7, 0), len(seq))
    seed_gc = _gc_content(seed)
    tail_gc = _gc_content(tail)
    seed_au = 1.0 - seed_gc
    tail_au = 1.0 - tail_gc
    return {
        f"{prefix}_seed_gc": seed_gc,
        f"{prefix}_tail_gc": tail_gc,
        f"{prefix}_seed_tail_gc_diff": seed_gc - tail_gc,
        f"{prefix}_seed_au": seed_au,
        f"{prefix}_tail_au": tail_au,
    }


def _positional_codes(seq: str, prefix: str) -> Dict[str, int]:
    seq = _normalize_sequence(seq)
    return {f"{prefix}_pos{idx + 1}_code": _base_code(base) for idx, base in enumerate(seq)}


def compute_sequence_features(df: pd.DataFrame, sequence_columns: Optional[Sequence[str]]) -> pd.DataFrame:
    sirnas = df["siRNA"].astype(str)
    mrnas = df["mRNA"].astype(str)
    extended = df.get("extended_mRNA")
    extended = extended.astype(str) if extended is not None else pd.Series([""] * len(df))

    rows = []
    for sirna_raw, mrna_raw, ext_raw in zip(sirnas, mrnas, extended):
        sirna = _normalize_sequence(sirna_raw)
        mrna = _normalize_sequence(mrna_raw)
        ext = _normalize_sequence(ext_raw)

        features: Dict[str, float] = {}
        features.update(_base_fraction_features(sirna, "sirna"))
        features.update(_base_fraction_features(mrna, "mrna"))
        features["sirna_mrna_gc_gap"] = (
            features.get("sirna_frac_G", 0.0)
            + features.get("sirna_frac_C", 0.0)
            - features.get("mrna_frac_G", 0.0)
            - features.get("mrna_frac_C", 0.0)
        )
        features.update(_seed_tail_features(sirna, "sirna"))
        features.update(_seed_tail_features(mrna, "mrna"))
        features["sirna_first_base_code"] = _base_code(sirna[:1])
        features["sirna_last_base_code"] = _base_code(sirna[-1:])
        features["mrna_first_base_code"] = _base_code(mrna[:1])
        features["mrna_last_base_code"] = _base_code(mrna[-1:])
        features.update(_pair_features(sirna, mrna))
        features.update(_base_fraction_features(ext, "extended"))
        features["extended_x_frac"] = features.get("extended_frac_X", 0.0)
        features["extended_gc_content"] = features.get("extended_frac_G", 0.0) + features.get("extended_frac_C", 0.0)
        features.update(_seed_tail_features(ext, "extended"))
        features["extended_first_base_code"] = _base_code(ext[:1])
        features["extended_last_base_code"] = _base_code(ext[-1:])
        features.update(_positional_codes(sirna, "sirna"))
        features.update(_positional_codes(mrna, "mrna"))
        rows.append(features)

    sequence_df = pd.DataFrame(rows, index=df.index)

    if sequence_columns is not None:
        sequence_df = sequence_df.reindex(columns=list(sequence_columns), fill_value=0.0)
        code_cols = [c for c in sequence_df.columns if c.endswith("_code")]
        if code_cols:
            sequence_df[code_cols] = sequence_df[code_cols].fillna(-1)

    return sequence_df


def load_artifacts() -> dict:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Representation artifact missing at {ARTIFACT_PATH}")
    artifact = joblib.load(ARTIFACT_PATH)
    required = {"feature_builder", "scaler", "feature_names"}
    missing = required.difference(artifact.keys())
    if missing:
        raise KeyError(f"Artifact missing keys: {sorted(missing)}")
    return artifact


def load_model() -> lgb.Booster:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file missing at {MODEL_PATH}")
    try:
        return lgb.Booster(params={"device_type": "gpu"}, model_file=str(MODEL_PATH))
    except (lgb.basic.LightGBMError, TypeError):
        return lgb.Booster(model_file=str(MODEL_PATH))


def prepare_features(df: pd.DataFrame, artifact: dict) -> np.ndarray:
    input_df = df.copy()
    for col in ["target", "numeric_label"]:
        if col in input_df.columns:
            input_df = input_df.drop(columns=col)

    sequence_columns = artifact.get("sequence_features")
    sequence_features = compute_sequence_features(input_df, sequence_columns)

    numeric_features = input_df.drop(columns=["siRNA", "mRNA", "extended_mRNA"], errors="ignore")
    feature_input = pd.concat([numeric_features, sequence_features], axis=1)

    feature_builder = artifact["feature_builder"]
    transformed = feature_builder.transform(feature_input)
    feature_names = artifact["feature_names"]
    transformed = transformed[feature_names]

    scaler = artifact["scaler"]
    scaled = scaler.transform(transformed)
    return scaled


def run_inference(input_path: Path, output_path: Path) -> None:
    data = pd.read_csv(input_path)
    required_cols = {"siRNA", "mRNA", "extended_mRNA"}
    missing_cols = required_cols.difference(data.columns)
    if missing_cols:
        raise ValueError(f"Input data missing required columns: {sorted(missing_cols)}")

    artifact = load_artifacts()
    features = prepare_features(data, artifact)
    booster = load_model()
    feature_names = artifact["feature_names"]
    preds = booster.predict(pd.DataFrame(features, columns=feature_names))
    pd.DataFrame({"prediction": preds}).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict siRNA knockdown efficacy")
    parser.add_argument("--input", required=True, help="Path to input CSV without target column")
    parser.add_argument("--output", required=True, help="Path to save predictions CSV")
    args = parser.parse_args()

    run_inference(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
