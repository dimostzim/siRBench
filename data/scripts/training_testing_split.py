from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt


def ks_min_split_save_plot(
    df,
    target,
    test_size=0.2,
    n_iters=500,
    train_filename="train_split.csv",
    test_filename="test_split.csv",
    random_state=42,
    n_bins=50
):

    def _hist_probs(values, bin_edges):
        counts, _ = np.histogram(values, bins=bin_edges)
        probs = counts.astype(float) / counts.sum()
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0)
        probs /= probs.sum() 
        return probs

    def compute_psi(p_ref, p_comp):
        return np.sum((p_comp - p_ref) * np.log(p_comp / p_ref))

    def compute_bhattacharyya(p, q):
        bc = np.sum(np.sqrt(p * q))   
        bc = np.clip(bc, 1e-12, 1.0)       
        return -np.log(bc)

    def compute_hellinger(p, q):
        return (1.0 / np.sqrt(2.0)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

    X = df.drop(columns=[target]).to_numpy()
    y = df[target].to_numpy()

    best_ks = np.inf
    best_split = None

    splitter = ShuffleSplit(
        n_splits=n_iters,
        test_size=test_size,
        random_state=random_state
    )

    for train_idx, test_idx in splitter.split(X):
        train_y = y[train_idx]
        test_y = y[test_idx]

        ks_value = ks_2samp(train_y, test_y).statistic

        if ks_value < best_ks:
            best_ks = ks_value
            best_split = (train_idx, test_idx)

    train_idx, test_idx = best_split
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    train_df.to_csv(train_filename, index=False)
    test_df.to_csv(test_filename, index=False)


    train_y = train_df[target].to_numpy()
    test_y  = test_df[target].to_numpy()

    all_values = np.concatenate([train_y, test_y])
    bin_edges = np.linspace(all_values.min(), all_values.max(), n_bins + 1)

    p_train = _hist_probs(train_y, bin_edges)
    p_test  = _hist_probs(test_y,  bin_edges)

    js_distance = jensenshannon(p_train, p_test, base=2.0)
    js_divergence = js_distance ** 2

    psi_value = compute_psi(p_train, p_test)

    w_distance = wasserstein_distance(train_y, test_y)

    bhatta_dist = compute_bhattacharyya(p_train, p_test)
    hellinger_dist = compute_hellinger(p_train, p_test)


    print("Train/test CSV files saved:")
    print(f"   → {train_filename}")
    print(f"   → {test_filename}\n")

    print("Distribution similarity metrics on target:", target)
    print(f"   KS statistic           : {best_ks:.6f}")
    print(f"   JS divergence          : {js_divergence:.6f}")
    print(f"   JS distance (sqrt(JS)) : {js_distance:.6f}")
    print(f"   PSI (train → test)     : {psi_value:.6f}")
    print(f"   Wasserstein distance   : {w_distance:.6f}")
    print(f"   Bhattacharyya distance : {bhatta_dist:.6f}")
    print(f"   Hellinger distance     : {hellinger_dist:.6f}")


    train_y_sorted = np.sort(train_y)
    test_y_sorted  = np.sort(test_y)

    train_cdf = np.arange(1, len(train_y_sorted) + 1) / len(train_y_sorted)
    test_cdf  = np.arange(1, len(test_y_sorted) + 1) / len(test_y_sorted)

    plt.figure(figsize=(8, 5))
    plt.plot(train_y_sorted, train_cdf, label="Train CDF")
    plt.plot(test_y_sorted,  test_cdf,  label="Test CDF")
    plt.title(f"CDF Comparison (KS = {best_ks:.6f})")
    plt.xlabel(target)
    plt.ylabel("CDF")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    metrics = {
        "ks_statistic": float(best_ks),
        "js_divergence": float(js_divergence),
        "js_distance": float(js_distance),
        "psi": float(psi_value),
        "wasserstein_distance": float(w_distance),
        "bhattacharyya_distance": float(bhatta_dist),
        "hellinger_distance": float(hellinger_dist),
    }

    return train_df, test_df, metrics


base_dir = Path(__file__).resolve().parents[1]
df = pd.read_csv(base_dir / "siRBench_train.csv")  # Use existing train set as base

train, test, metrics = ks_min_split_save_plot(
    df,
    target="efficacy",
    test_size=0.09,
    n_iters=2168,
    train_filename=base_dir / "siRBench_train.csv",
    test_filename=base_dir / "siRBench_test.csv",
    n_bins=50
)

print(metrics)
