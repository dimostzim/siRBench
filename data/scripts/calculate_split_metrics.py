from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon

DATA_DIR = Path(".")

files = {
    "training": DATA_DIR / "siRBench_train_split.csv",
    "validation": DATA_DIR / "siRBench_val_split.csv",
    "testing": DATA_DIR / "siRBench_test.csv",
    "leftout": DATA_DIR / "siRBench_leftout.csv",
    "hela": DATA_DIR / "siRBench_hela.csv",
}

pairs = [
    ("training", "validation"),
    ("training", "leftout"),
    ("training", "testing"),
    ("training", "hela"),
]

TARGET_COL = "efficiency"
N_BINS = 10
BIN_RANGE = (0.0, 1.0)
EPS = 1e-10

def load_target(csv_path, target_col=TARGET_COL):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in {csv_path}")
    values = pd.to_numeric(df[target_col], errors="coerce")
    values = values.dropna().to_numpy(dtype=float)
    return values

def get_histogram_probabilities(values, bins, eps=EPS):
    counts, _ = np.histogram(values, bins=bins)
    probabilities = counts / counts.sum()
    probabilities = np.clip(probabilities, eps, None)
    probabilities = probabilities / probabilities.sum()
    return probabilities

def hellinger_distance(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

def population_stability_index(p, q):
    return np.sum((p - q) * np.log(p / q))

def calculate_distribution_metrics(values_a, values_b, bins):
    p = get_histogram_probabilities(values_a, bins)
    q = get_histogram_probabilities(values_b, bins)
    ks_result = ks_2samp(values_a, values_b)
    js_distance = jensenshannon(p, q, base=2.0)
    js_divergence = js_distance ** 2
    metrics = {
        "KS distance": ks_result.statistic,
        "KS p-value": ks_result.pvalue,
        "JS distance": js_distance,
        "JS divergence": js_divergence,
        "PSI": population_stability_index(p, q),
        "Hellinger Distance": hellinger_distance(p, q),
        "Wasserstein Distance": wasserstein_distance(values_a, values_b),
    }
    return metrics


data = {
    name: load_target(path)
    for name, path in files.items()
}

bins = np.linspace(BIN_RANGE[0], BIN_RANGE[1], N_BINS + 1)
results = []

for dataset_a, dataset_b in pairs:
    values_a = data[dataset_a]
    values_b = data[dataset_b]
    metrics = calculate_distribution_metrics(values_a, values_b, bins)
    row = {
        "Pair": f"{dataset_a}-{dataset_b}",
        "n_A": len(values_a),
        "n_B": len(values_b),
        **metrics,
    }
    results.append(row)

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("efficacy_distribution_similarity_metrics.csv", index=False)