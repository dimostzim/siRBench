import pandas as pd

initial_csv = "siRBench_hela.csv"
remaining_csv = "siRBench_leftout.csv"
output_csv = "siRBench_leftout_dropped.csv"

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

