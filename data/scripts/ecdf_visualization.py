import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ecdf(values):
    """Compute ECDF (x, y)."""
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

train_df = pd.read_csv("siRBench_train.csv")
test_seen_df = pd.read_csv("siRBench_test.csv")
test_loco_df = pd.read_csv("siRBench_left_out_test.csv")  # HeLa

label_col = "efficacy" 


x_train, y_train = ecdf(train_df[label_col].values)
x_seen, y_seen = ecdf(test_seen_df[label_col].values)
x_loco, y_loco = ecdf(test_loco_df[label_col].values)


plt.figure(figsize=(7, 5))

plt.step(x_train, y_train, where="post", label="Train")
plt.step(x_seen, y_seen, where="post", label="Test (seen cell lines)")
plt.step(x_loco, y_loco, where="post", label="Test (left-out cell line)")

plt.xlabel("Label value")
plt.ylabel("CDF")
plt.title("CDF comparison across train and test sets")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
