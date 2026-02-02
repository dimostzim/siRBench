from pathlib import Path

from scipy.stats import ks_2samp
import pandas as pd

base_dir = Path(__file__).resolve().parents[1]
train_df = pd.read_csv(base_dir / "siRBench_train.csv")
left_out_df = pd.read_csv(base_dir / "leftout" / "siRBench_leftout.csv")

label_col = "efficiency"

x = train_df[label_col].dropna().values
y = left_out_df[label_col].dropna().values

ks_stat, p_value = ks_2samp(x, y)

print(f"KS statistic: {ks_stat:.6f}")
print(f"KS p-value : {p_value:.6e}")
