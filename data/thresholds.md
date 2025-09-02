# Cell-line Thresholds for siRBench

This note summarizes how we derived a single “binary” decision threshold (on normalized efficacy in [0,1]) for each cell line in siRBench.

See `data/thresholds_per_cell_line.csv` for the final table.

## Summary of Method

- All siRBench efficacies are already normalized to [0,1]. We define a binary label as `1` if `efficacy >= threshold`, else `0`.
- We set thresholds by finding the label separation point (midpoint between the highest negative and lowest positive) wherever a clean split exists, using the datasets the mix set originates from; for a remaining cell line (hep3b/Simone) we chose a robust aggregate from other lines.

## How Each Threshold Was Obtained

- h1299 (Huesken): Perfect separation in `hu.csv` on normalized label; threshold = midpoint between max(negatives) and min(positives) = `0.521626`.
- hela (Takayuki): Perfect separation in `taka.csv`; threshold = midpoint = `0.714512`.
- halacat, hek293, hek293t, t24 (from mix set): We matched each `mix.csv` (siRNA, mRNA) pair to siRBench by exact siRNA and RNA-form mRNA as a substring of `extended_mRNA` (T→U). For each cell line subset we computed:
  - `neg_max` = max label where `y=0`, `pos_min` = min label where `y=1`.
  - Threshold = `(neg_max + pos_min)/2` (all four had perfect separation).
  - Results: halacat `0.550000`, hek293 `0.787964`, hek293t `0.792488`, t24 `0.921053`.
- hep3b (Simone): No direct mix-derived split. We adopted a robust baseline equal to the median of the other non-Huesken/non-Takayuki lines’ thresholds (halacat, hek293, hek293t, t24): median(`0.55, 0.787964, 0.792488, 0.921053`) = `0.790226`.

## Final Thresholds

- h1299: 0.521626
- hela: 0.714512
- hek293: 0.787964
- hek293t: 0.792488
- t24: 0.921053
- halacat: 0.550000
- hep3b: 0.790226 (median of other non-Huesken/non-Takayuki lines)

These thresholds can be used to create a `binary` column: `binary = 1 if efficacy >= threshold[cell_line] else 0`.

