#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

from tabpfn import TabPFNRegressor

DROP_COLUMNS = ["siRNA", "mRNA", "extended_mRNA", "binary", "source", "cell_line"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    regressor = TabPFNRegressor(
        n_estimators=32,
        softmax_temperature=9.6,
        device="cuda",
        random_state=42,
    )

    RESULTS_DIR.mkdir(exist_ok=True)

    for split in ["train_90", "val_10", "test"]:
        csv_path = DATA_DIR / f"siRBench_{split}.csv"
        df = pd.read_csv(csv_path)
        X = df.drop(columns=["efficacy"] + DROP_COLUMNS)
        y = df["efficacy"]

        if split == "train_90":
            print("Fitting model on train set...")
            regressor.fit(X, y)
        elif split == "val_10":
            print("Getting predictions on validation set...")
        else:  # test
            print("Getting predictions on test set...")

        preds = regressor.predict(X)
        loss = ((y - preds) ** 2).mean()
        pcc = pearsonr(y, preds)[0]

        pd.DataFrame({"true": y, "pred": preds}).to_csv(
            RESULTS_DIR / f"{split}_predictions.csv", index=False
        )

        print(f"{split:>8}: Loss = {loss:.4f}, PCC = {pcc:.4f}")


if __name__ == "__main__":
    main()
