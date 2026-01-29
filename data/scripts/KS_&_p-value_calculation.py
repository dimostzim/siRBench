from scipy.stats import ks_2samp
import pandas as pd

train_df = pd.read_csv("siRBench_train.csv")
left_out_df = pd.read_csv("siRBench_leftout.csv")

label_col = "efficacy"

x = train_df[label_col].dropna().values
y = left_out_df[label_col].dropna().values

ks_stat, p_value = ks_2samp(x, y)

print(f"KS statistic: {ks_stat:.6f}")
print(f"KS p-value : {p_value:.6e}")
