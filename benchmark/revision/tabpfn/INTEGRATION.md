# TabPFN-3.5 benchmark integration

Prepared 22 September 2026. This document describes the integration contract; it
does not report TabPFN outer-test or HeLa results. The artifact audit checks held-out
membership, labels and finite predictions without scoring them; the subsequent
common evaluator computes held-out metrics only after the complete matrix is sealed.
Keep the existing six-method release and validation-only pilot unchanged.

## Frozen evaluation contract

Canonical workspace on biogemt-4:
`/SCRATCH/dtzim01/sirbench-revision-20260921` (called `$R` below).

The existing analysis code is
`$R/siRBench/benchmark/revision/evaluate_predictions.py`. It already accepts
multiple `--run-index` arguments and arbitrary additional method identifiers.
Its validation and statistical procedures require no modification to include
TabPFN. In particular, do not modify `assemble_primary_index.py`, which
intentionally validates the original six-method, 180-run matrix.

Preserve these inputs:

| Input | Canonical path relative to `$R` |
|---|---|
| Original six methods, 180 verified runs | `evaluation/primary-index-v1/run_index.csv` |
| Seven reference methods, including two separately labeled historical iScore variants | `evaluation/baselines-v1/predictions.csv` |
| Corrected records and feature columns | `datasets/corrected-v1/records_features.csv` |
| Frozen protocol and memberships | `evaluation/protocol-v1/` |
| Existing complete analysis | `evaluation/primary-analysis-v1/` |

Use distinct method identifiers for the two 3.5 variants:

- `tabpfn35_frozen`: fixed foundation weights; seed controls preprocessing and
  ensemble randomness, not gradient training.
- `tabpfn35_finetuned`: foundation weights updated on the fold training data;
  validation selects the checkpoint under the locked fine-tuning policy.

Each variant needs all 30 identities:
`axis in {grouped, random} × fold in {0,1,2,3,4} × training_seed in {0,1,2}`.
Together they add 60 runs to the unchanged original 180. The current grouped-only
validation pilot is not a complete benchmark matrix and cannot supply a main
test result by itself.

Every run predicts exactly the corresponding outer-test IDs and all 1,047 HeLa
IDs. The grouped test counts for folds 0–4 are 656, 669, 662, 500 and 564; each
random test has 275 rows. The aligned 896-row HeLa subset is derived automatically
from the full HeLa predictions by the existing evaluator. It requires no extra
training or inference.

Keep the 176-column pilot representation fixed for both variants: 76 positional
guide one-hot features and 100 approved sequence/thermodynamic features. The
features describe the guide/central duplex, not the context flanks. Source, cell
line, target identity, group identity and labels are excluded from predictors.

## Prediction/index schema

An individual prediction CSV has exactly one row per evaluation record, with:

```text
record_id,label,pred_label
```

`id` is also accepted in place of `record_id`. Predictions must be finite. Labels
must equal the frozen record labels within absolute tolerance 1e-12. Do not clip,
calibrate, reorder without retaining IDs, or silently drop failed predictions.

A separate TabPFN `run_index.csv` contains these columns:

```text
tool,axis,fold,training_seed,run_dir,test_predictions,hela_predictions,train_meta,status,test_predictions_sha256,hela_predictions_sha256,train_meta_sha256
```

Paths are absolute canonical node-4 paths. Each artifact path must belong to
exactly one run identity. `status` is `verified` only after artifact auditing.
The three hash columns seal the exact file bytes. `train_meta` points to JSON
containing at least top-level `seed`, equal to `training_seed`. If a nested
`config.seed` or `configuration.seed` is present, it must agree too.

Additional train metadata should retain the input and feature-manifest hashes,
foundation checkpoint hash, package versions, code hashes, feature allowlist,
fit mode, seed interpretation, fixed inference batching/precision, model-save
hashes and validation selection policy. Fine-tuned runs should also retain the
actual epochs, optimizer steps, selected epoch, learning rate and training history.
These enrich provenance; the generic evaluator directly checks seed consistency
and the sealed artifact bytes.

The index does not establish that a model was correctly trained: verify saved
model reloads and held-out prediction identities before sealing the index. The
frozen pilot detected small batch-dependent numerical differences. The main
inference convention is one call for the full evaluation cohort, eight estimators,
and the released fingerprint behavior; keep this fixed and document the numerical
checks rather than selecting behavior on held-out performance.

## One common numerical analysis

Run only after the training/selection and inference policy are locked, all 60
TabPFN identities are complete, and the new index is independently verified.
The example reserves fresh directories; adjust only the new index/output paths
to the runner's actual destinations.

First run the independent artifact audit in the TabPFN environment. It refuses
an incomplete 30-identity matrix and ambiguous multiple successful attempts,
preserves failed attempts, and publishes the 60-row index only after all checks
pass. The output defaults to the existing benchmark directory and refuses to
overwrite an earlier index/manifest:

```bash
R=/SCRATCH/dtzim01/sirbench-revision-20260921
"$R/evaluation/tabpfn-v1/.venv/bin/python" \
  "$R/evaluation/tabpfn-v1/code/audit_benchmark.py" \
  --root "$R" --runs "$R/evaluation/tabpfn-benchmark-v1/runs"
```

The common numerical evaluation uses the **original benchmark environment**,
`$R/.venv`, to retain the original NumPy/pandas/SciPy/scikit-learn versions.
The newer TabPFN environment remains appropriate for artifact/provenance checks
and plotting, but must not silently replace the numerical evaluation environment.

```bash
R=/SCRATCH/dtzim01/sirbench-revision-20260921
"$R/.venv/bin/python" \
  "$R/siRBench/benchmark/revision/evaluate_predictions.py" \
  --records "$R/datasets/corrected-v1/records_features.csv" \
  --protocol "$R/evaluation/protocol-v1" \
  --baselines "$R/evaluation/baselines-v1/predictions.csv" \
  --run-index "$R/evaluation/primary-index-v1/run_index.csv" \
  --run-index "$R/evaluation/tabpfn-benchmark-v1/run_index.csv" \
  --output "$R/evaluation/primary-analysis-with-tabpfn-v1" \
  --bootstrap-draws 2000
```

Training diagnostics are separate and use only the audited training/validation
summary. They report initial versus selected validation R², completed and
selected epochs, and how often initial foundation weights remain selected:

```bash
"$R/evaluation/tabpfn-v1/.venv/bin/python" \
  "$R/evaluation/tabpfn-v1/code/summarize_benchmark_training.py" \
  --audit "$R/evaluation/tabpfn-benchmark-v1" \
  --output "$R/evaluation/tabpfn-benchmark-v1/training-summary-v1"
```

Those validation improvements are selection-optimistic diagnostics. They are
not held-out performance estimates and are not used to revise the locked policy.

This writes the existing output schema, now containing 15 methods: six published
siRNA predictors, two TabPFN variants, five primary reference baselines and two
historical iScore references. The five primary baselines are training mean,
guide ridge, thermodynamic ridge, combined ridge and calibrated Ui-Tei score.

- `replicate_metrics.csv`: pooled and source/cell-line metrics for each replicate.
- `metric_summary.csv`: descriptive replicate means/SDs.
- `replicate_dispersion.csv`: within-fold seed and between-fold dispersion.
- `fold_rankings.csv`: descriptive fold/seed rankings, excluding historical iScore.
- `macro_metrics.csv`: per-source and per-cell-line macro metrics and finite counts.
- `primary_group_bootstrap.csv`: grouped-test and full/aligned-HeLa point estimates
  with 95% intervals.
- `primary_paired_differences.csv`: paired differences for all method pairs.
- `analysis_manifest.json`: versions, input/code hashes and statistical conventions.

There are thirteen eligible primary methods. Within an individual fold the
training-mean baseline has constant predictions, so its Pearson correlation is
undefined: expect twelve finite Pearson ranks and thirteen finite R² ranks.
Check the actual rank table before reporting those counts. Grouped out-of-fold
training-mean predictions combine different fold-specific means, so the pooled
correlation need not be undefined.

Grouped test metrics pool five out-of-fold predictions into 3,051 unique records
within each seed, compute each metric, then average the three seed metrics.
HeLa metrics are calculated for each of the 15 fold/seed predictors, then averaged;
predictions are not averaged across fold/seed fits; TabPFN retains its eight
internal inference ensemble members. The 2,000 bootstrap draws resample the same clusters
for every method: 43 target/guide components for grouped test and 45 target groups
for HeLa. Intervals are conditional on the fitted models and frozen partitions.
Paired comparisons are exploratory and unadjusted for multiplicity. Random-split
results retain the existing descriptive treatment; the evaluator does not claim
independent-fold confidence intervals for overlapping random tests.

Useful predeclared comparisons are fine-tuned versus frozen 3.5, each 3.5 variant
versus combined ridge using the same 176 inputs, and each variant versus the six
published siRNA methods. Report all methods even if fine-tuning is unsuccessful.

Before rendering, verify that every original numerical row, input hash and
analysis setting is unchanged, and independently recompute the two new pooled
grouped-test R² values from the saved predictions:

```bash
"$R/.venv/bin/python" \
  "$R/evaluation/tabpfn-v1/code/verify_expanded_analysis.py" \
  --original "$R/evaluation/primary-analysis-v1" \
  --expanded "$R/evaluation/primary-analysis-with-tabpfn-v1" \
  --tabpfn-index "$R/evaluation/tabpfn-benchmark-v1/run_index.csv" \
  --output "$R/evaluation/primary-analysis-with-tabpfn-v1/invariance_verification.json"
```

The original within-fold ranks may change when two comparators are added; this
check preserves the original predictions, metrics, intervals and paired contrasts.
The separate validation-only portability control is recorded under
`evaluation/tabpfn-benchmark-v1/portability-v2/`. It measures fresh-process same-node
and returned-worker reload differences without replacing any original prediction.
Same-device acceptance checks and cross-node/runtime sensitivity are reported
separately; this control does not establish which hardware or preprocessing
operation causes a numerical difference.

## Rendering and release checks

`render_primary_results.py` is deliberately fixed to the original six-method
index and thirteen-method analysis. Do not invoke it unchanged on the expanded
analysis: its guards correctly reject extra methods. The separate
`render_benchmark_results.py` adds two display rows, verifies both the original
180-run and new 60-run indexes, and distinguishes preprocessing/ensemble
replicates from gradient-training replicates in the caption. It does not modify
the original renderer or figures. After the common analysis succeeds, run:

```bash
"$R/evaluation/tabpfn-v1/.venv/bin/python" \
  "$R/evaluation/tabpfn-v1/code/render_benchmark_results.py" \
  --analysis "$R/evaluation/primary-analysis-with-tabpfn-v1" \
  --original-index "$R/evaluation/primary-index-v1/run_index.csv" \
  --tabpfn-index "$R/evaluation/tabpfn-benchmark-v1/run_index.csv" \
  --output "$R/manuscript-revision/tabpfn/benchmark-results/figures"
```

The output includes PDF/PNG/SVG figures, machine-readable interval tables,
Markdown/LaTeX tables for grouped test and full HeLa, captions, alt text and a
hash manifest. Both TabPFN variants are retained in fixed order. The renderer
does not recompute or select numerical results. Its validation/plot tests use
synthetic intervals only:

```bash
"$R/evaluation/tabpfn-v1/.venv/bin/python" -m pytest \
  "$R/evaluation/tabpfn-v1/code/test_benchmark_results.py" -q
```

Before incorporating expanded results into the manuscript or rebuttal:

1. Check every new prediction against frozen membership, labels and SHA256s;
   confirm both 30-run variant matrices and unique artifact paths.
2. Verify saved-model reload predictions, recorded checkpoint choice, actual
   training history and fixed inference behavior.
3. Recompute representative metrics independently and confirm all original
   method point estimates and intervals are unchanged in the expanded analysis.
   Their rankings can change because two legitimate comparands were added.
4. Render all-method tables/plots and visually inspect labels and intervals.
   Retain the historical iScore distinction and undefined correlations.
5. Copy every returned worker artifact into node 4 SCRATCH and record hashes
   before cleaning worker copies. Archive code, protocol and training metadata;
   handle licensed foundation weights according to their accepted terms.

The benchmark claim is a comparison under this frozen representation/protocol.
Validation development and numerical improvement alone do not establish a
general state-of-the-art siRNA model.
