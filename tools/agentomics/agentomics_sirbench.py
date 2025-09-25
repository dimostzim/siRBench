#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

from inference import load_artifacts, load_model, prepare_features

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPLITS = ["train_90", "val_10", "test"]


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    artifact = load_artifacts()
    booster = load_model()
    feature_names = artifact["feature_names"]

    for split in SPLITS:
        csv_path = DATA_DIR / f"siRBench_{split}.csv"
        df = pd.read_csv(csv_path)
        features = prepare_features(df, artifact)
        preds = booster.predict(pd.DataFrame(features, columns=feature_names))

        output_path = RESULTS_DIR / f"{split}_predictions.csv"
        pd.DataFrame({"true": df.get("efficacy"), "pred": preds}).to_csv(output_path, index=False)

        if "efficacy" in df.columns:
            loss = ((df["efficacy"] - preds) ** 2).mean()
            pcc = pearsonr(df["efficacy"], preds)[0]
        else:
            loss = float("nan")
            pcc = float("nan")

        if split == "train_90":
            print(f"{split:>8}: Loss = {loss:.4f}")
        elif split == "val_10":
            print(f"{split:>8}: Loss = {loss:.4f}, PCC = {pcc:.4f}")
        else:
            print(f"{split:>8}: PCC = {pcc:.4f}")


if __name__ == "__main__":
    main()
