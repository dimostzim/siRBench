# TabPFN-3.5 benchmark extension

This directory adds separately reported frozen and gradient-fine-tuned
TabPFN-3.5 predictors to the revised siRBench evaluation. It uses the existing
frozen protocol and combined ridge representation, not assay metadata.
The original six-method index and analysis remain separate inputs.

Read `BENCHMARK_POLICY.md` for the fixed experimental design and `INTEGRATION.md`
for complete audit, analysis and rendering commands. The earlier v2/3.5
validation-only pilot is separate from this main experiment.

## Install with uv

Use [UV_REPRODUCE.md](UV_REPRODUCE.md) for the pinned `pyproject.toml` /
`uv.lock` environment, training, saved-state prediction and feature-input
commands. [UV_VERIFICATION.md](UV_VERIFICATION.md) reports fresh-environment
checks and cross-device numerical limits. The original run policy and sealed
results are unchanged.

## Verified completion

The full experiment is complete: 30 gradient fine-tuning runs and 30 frozen
context fits, with an independently sealed sixty-row index. Common evaluation
used the original numerical environment and preserved every original result,
input hash and analysis setting. Both new pooled grouped-test R² values were
also independently recomputed from saved predictions. The two variants remain
separate regardless of their relative performance; no broad state-of-the-art
claim is made.

Canonical outputs relative to `$R` are:

- `evaluation/tabpfn-benchmark-v1/run_index.csv` and `manifest.json`: sealed fits and artifact audit.
- `evaluation/primary-analysis-with-tabpfn-v1/`: complete fifteen-method results and `invariance_verification.json`.
- `manuscript-revision/tabpfn/benchmark-results/REPORT.md`: numerical interpretation and limits.
- `manuscript-revision/tabpfn/benchmark-results/{figures,training,supplement,portability}/`: figures, training evidence, tables and validation-only reload controls.

The grouped frozen/fine-tuned R² values are 0.3967/0.3985; the paired difference
interval includes zero. Full-HeLa R² improves with fine-tuning while Pearson r
slightly decreases, and random-partition results favor ENsiRNA. All original
methods, failed-attempt evidence and pilot results are preserved. The worker
scratch roots were removed only after every returned artifact and the complete
canonical matrix passed verification. Follow the commands below in fresh output
directories; do not overwrite the completed reference experiment.

## Required workspace and runtime

The reference workspace root is
`/SCRATCH/dtzim01/sirbench-revision-20260921` (`$R` below). A relocated copy must
retain its workspace-relative layout:

- `siRBench/benchmark/revision/`: shared baselines, metric and artifact helpers.
- `datasets/corrected-v1/records_features.csv` and its feature manifest.
- `evaluation/protocol-v1/`: frozen train/validation/test CSVs, memberships and
  manifest; all 1,047 HeLa records remain separate.
- `evaluation/tabpfn-v1/models/v3.5/tabpfn-v3.5-20260909.safetensors`: official
  checkpoint, obtained through the model's accepted access route.

The runner checks the foundation, protocol, feature-manifest and partition
hashes before fitting. The checkpoint SHA256 is
`ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`.
Foundation weights and private generated run artifacts are not included in this
source-code directory.

Use Python 3.12.3, TabPFN 9.0.0 at source commit
`9393b12a46bfc32369a89a53f6c578d22e41faba`, PyTorch 2.8.0+cu126 and the packages
recorded in `runtime.lock.txt`. This lock is a record of the actual runtime,
including its local TabPFN source path; install that pinned source explicitly
when reproducing at another location. The demonstrated full-data training setup
fits the 16 GB RTX A4000 and 24 GB A10 used in the experiment.

## Run one pair

Activate the isolated TabPFN runtime, set BLAS/OpenMP threads to four, and run:

```bash
python run_benchmark.py \
  --root "$R" \
  --axis grouped --fold 0 --seed 0 \
  --output "$R/evaluation/tabpfn-benchmark-v1/runs/grouped/fold_0/seed_0/attempt_2"
```

The output directory must not already exist. It contains separate
`tabpfn35_frozen/` and `tabpfn35_finetuned/` artifacts, a policy lock, two-row
prediction index and completion seal. The runner saves validation/test/full-HeLa
predictions, fitted training contexts, fine-tuned selected weights, training and
validation histories, effective configuration, hashes and reload checks.
Held-out cohorts are scored only after fitting and checkpoint selection.

Run all identities in `axis={grouped,random}`, `fold={0,1,2,3,4}` and
`seed={0,1,2}`. This produces 30 gradient fine-tuning runs and 30 frozen context
fits. `launch_benchmark.py` is the site-specific three-worker queue used for the
reference experiment; its root and node assignments are documented in its
manifest. It preserves failed attempts and does not use upstream automatic
checkpoint resume.

## Audit and analysis

After all 30 identities and both variants are available at their canonical
paths, `audit_benchmark.py` verifies the complete matrix and produces a separate
60-row index. It refuses incomplete or ambiguous duplicate successes.
`summarize_benchmark_training.py` reports training/selection histories.

The common numerical analysis uses the original siRBench analysis environment,
not the newer TabPFN runtime. `INTEGRATION.md` gives its exact command with the
unchanged 180-row original index, the new 60-row index and the original reference
baselines. `verify_expanded_analysis.py` requires every original numerical
result and analysis setting to remain unchanged, and independently recomputes
both new grouped R² estimates. `render_benchmark_results.py` creates expanded
tables, plots, captions and alt text.

Focused tests are `test_benchmark.py`, `test_audit_benchmark.py`,
`test_benchmark_results.py`, `test_expanded_analysis.py` and
`test_expanded_supplement.py`. The separate
validation-only pilot has its own tests. Failed production attempts, probes and
pilot fits are not counted as successful main benchmark training runs.
