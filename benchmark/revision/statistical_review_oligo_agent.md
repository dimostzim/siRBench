# Independent statistical and baseline review

Reviewed `evaluate_predictions.py`, `baselines.py`, their focused tests, and `evaluation/baseline-analysis-v2` on 2026-09-21 while comparator training continued. No runtime or frozen input files changed for this review.

## Findings

1. **Metadata coverage can silently reduce an evaluation cohort.** `add_metadata` uses inner joins (currently lines 81–82). A two-row prediction example with one missing group record returns one row without error. Prediction validation occurs before these joins. Use coverage-preserving joins and reject missing grouping/stratum metadata before metric calculation. The actual frozen group table covers all 4,098 records, so this does not invalidate the current baseline analysis.
2. **The analysis manifest does not fully identify comparator inputs.** It hashes the run-index CSV, but not the prediction files that index references; changing a prediction file leaves the recorded input hashes unchanged. It also hashes the protocol manifest without checking the actual groups/membership files against it. Hash every consumed prediction CSV and actual grouping/membership files, or verify their declared hashes at ingestion. This is a reproducibility gap, with no evidence that current files are inconsistent.
3. **Historical iScore overlap must remain visible downstream.** Both fixed and recalibrated iScore retain supervised overlap with 2,361 benchmark Huesken guides; affine recalibration does not remove that overlap. Initially they entered ordinary rank outputs. The current code now labels them `historical_reference`, excludes them from primary ranks, and marks pairwise comparisons involving them as nonprimary. That addresses the code-level issue; preserve these distinctions in manuscript tables and interpretation. Earlier baseline-analysis-v2 outputs predate these role columns.
4. **Focused statistical regression tests would protect the central inference.** Existing tests check constants, pairing, IDs/labels, missing runs, baseline train-only fitting, classical rules, and historical-reference ranks. Add explicit examples for complete OOF pooling, averaging replicate metrics rather than averaging predictions, equal HeLa fold/seed weighting, deterministic comparand duplication without counting extra fits, and metadata coverage. These are coverage recommendations; independent checks below found no current mathematical discrepancy.

## Verified behavior

- Independently recalculated all seven methods × six grouped OOF metrics with SciPy/scikit-learn: all 42 agree within 1e-12. Evidence: `audit/oligo_bert/statistical_review_metric_checks.csv`.
- All 378 paired point differences in baseline-analysis-v2 equal the corresponding left estimate minus right estimate.
- Grouped OOF evaluation has 3,051 non-HeLa rows, 47 target groups, and 43 target-plus-similar-guide bootstrap components. Full HeLa uses 1,047 rows and 45 conservative target clusters.
- Cluster draws are shared across methods. Whole clusters are sampled with replacement. Metrics are calculated for each fitted prediction replicate and then averaged; predictions are not ensembled.
- Deterministic baselines have one fit per partition. Repeating their rows only when building comparison ranks does not inflate their metric replicate counts.
- Constant predictions correctly produce undefined correlations. All 2,000 draws are finite except the expected HeLa correlations for the constant training-mean baseline. Pooled grouped OOF means can vary between folds and therefore have finite correlation; this is a property of pooled fold predictions, not a constant-metric bug.
- Confidence intervals are explicitly conditional on fitted models and frozen partitions. Descriptive seed/split SDs are not independent-sample standard errors. Pairwise intervals are exploratory and unadjusted.
- Ridge scalers and coefficients are fitted on training data only; alpha is selected on validation R². The 100 thermodynamic features inspected contain no source/cell-line covariates. Classical iScore published examples and Ui-Tei functional rules are covered by focused checks.

## Interpretation limits

The HeLa estimate averages predictions scored separately from models trained on each grouped training fold; it does not describe one predictor trained on all non-HeLa records. Describe this clearly. Conditional cluster-bootstrap intervals quantify held-out cluster sampling uncertainty, not full model-search, training-seed, or split-design uncertainty. These limits are already represented in the analysis manifest and should remain visible in the paper.
