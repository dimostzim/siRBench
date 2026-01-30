from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parents[1]
leftout_dir = base_dir / "leftout"
initial_csv = leftout_dir / "siRBench_hela.csv"
remaining_csv = leftout_dir / "siRBench_leftout.csv"
output_csv = leftout_dir / "siRBench_leftout_dropped.csv"

df_initial = pd.read_csv(initial_csv)
df_remaining = pd.read_csv(remaining_csv)

df_remaining = df_remaining[df_initial.columns]

df_erased = (
    df_initial
    .merge(
        df_remaining.drop_duplicates(),
        how="left",
        indicator=True
    )
    .query('_merge == "left_only"')
    .drop(columns="_merge")
)

df_erased.to_csv(output_csv, index=False)

