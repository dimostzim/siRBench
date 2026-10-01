# OligoFormer and BERT-siRNA wrapper audit

Audit date: 2026-09-21. The author approved the common training policy below.
All final artifacts belong on biogemt-4; biogemt-5 is temporary GPU staging only.
Native siRBench ensemble and Agentomics work are deferred.

## Pinned source comparison

| Property | OligoFormer | BERT-siRNA |
|---|---|---|
| Official repository | lulab/OligoFormer | ChengkuiZhao/siRNABERT |
| Commit | e2f53ad63387bbe166bf123949151e2bc9bf6ec3 | 00c849af12bbfad59efd106b6a28b0dfc766ac33 |
| Architecture | Imported upstream `Oligo` with defaults: embedding128, LSTM32, 8heads, 1layer, left/right context19 | Wrapper reproduces official DNABERT 6-mer CLS →128→128→64→1 MLP, ReLUs, dropout0.1, sigmoid output |
| Input | Guide19, target context57, 24 thermodynamic features, frozen RNA-FM embeddings; matches source default extent | Guide19 converted to DNA6-mers; max token length16; matches source training tokenization |
| Optimizer | Adam lr1e-4; ExponentialLR gamma0.999 | Adam lr5e-5, weight decay0.01 |
| Batch size | 16 | 100 |
| Official training | Up to200epochs; patience30; checkpoint requires lower continuous validation MSE AND higher binary AUC | 30 epochs; no early stopping or saved-best-checkpoint implementation in released training script |
| Historical common wrapper | Up to100epochs; patience 20; validation R² checkpoint selection | Up to100epochs; patience 20; validation R² checkpoint selection |

The OligoFormer option called `weight_decay` is used as the exponential scheduler's
gamma in the official source and wrapper, not as Adam weight decay. BERT validation
in the wrapper explicitly uses eval mode, whereas the released demonstration
training script does not switch model modes. Neither source's test labels will be
used for model selection in the revised benchmark.

## Confirmed corrections

- OligoFormer `eval_epoch` used binary threshold labels to calculate the R² used
  for checkpoint selection. It now uses continuous efficacy. Binary labels remain
  appropriate only for its auxiliary AUC. This bug concerns checkpoint selection;
  the wrapper's final regression metrics already used continuous test labels.
- Both preparation wrappers now propagate input `record_id` to prepared `id` when
  the legacy `id` column is absent. Predictions already export prepared `id` in
  input order. Explicit `--id-col` and historical fallback behavior are retained.
- Training metadata now records the stopping metric, selected epoch and, for BERT,
  optimizer-relevant run settings/seed previously omitted.

Runtime regression evidence on node 4:
`audit/oligo-bert-checks-before.txt` shows perfect efficacy predictions wrongly
returning R²=0.79. `audit/oligo-bert-checks-after.txt` shows the fixed R²=1 and both
record-ID propagation checks passing. The test uses real PyTorch tensors and
unequal batch sizes, not source-text assertions.

## Remaining validation

Docker builds, pretrained-asset inventory, and real preparation/training/reload
smokes are complete. The approved 30-run matrix per method is now running on
node 5 under a node 4 controller; establish runtime and validate completed results. Store
predictions, checkpoints, logs and manifests centrally and verify checksums before
removing temporary node 5 data. No scientific model results are claimed yet.

## BERT-siRNA identity cross-check

The manuscript bibliography cites Xu, Xu, Xie, Zhao, Yu and Feng, Gene 910:148330
(2024). [PubMed 38431236](https://pubmed.ncbi.nlm.nih.gov/38431236/) and the
[publisher record](https://www.sciencedirect.com/science/article/pii/S0378111924002117)
confirm this title, author list and DOI 10.1016/j.gene.2024.148330. The
[author repository](https://github.com/ChengkuiZhao/siRNABERT) README explicitly
identifies the same paper and authors. This establishes the author-repository
identity; the publisher full text was unavailable (HTTP403), so its direct
repository backlink could not be checked. A newer secondary review lists
`nxu1/BERT-siRNA`, but that URL returned404/nonpublic and was not adopted as an
authoritative replacement.

## Approved main training policy

The author approved at most 100 epochs with patience 20 on best-so-far continuous
validation R², preserving released architecture/optimizer/learning-rate/scheduler.
OligoFormer common-mode patience now stops after 20 non-improving epochs; the
explicit original-params mode preserves the upstream greater-than boundary.
BERT's legacy original mode used30 epochs/no early stopping with wrapper-best
validation-loss checkpoint selection. The revision's separately staged schedule
control instead retains the final epoch checkpoint, as documented below.
Targeted original-schedule sensitivity is secondary, rather than a second full matrix.

All 2,361 official OligoFormer Huesken 24-feature vectors agree with the wrapper to
floating-point precision (maximum absolute difference 5.7e-14). The real revision
container passed RNA-FM preparation,two-epoch training, checkpoint reload and
prediction export with persistent record IDs on a tiny smoke fixture. Those smoke
metrics are not benchmark evidence.

Shared runner now forwards OMP_NUM_THREADS/MKL_NUM_THREADS(default4) and
TF_NUM_INTRAOP_THREADS(default4)/TF_NUM_INTEROP_THREADS(default2), avoiding CPU
oversubscription while allowing explicit host overrides.


## Execution and asset inventory

OligoFormer image: `sha256:54c0e836510d00637d0b2d86c39843528f5944b8bdb1b1e3ad9fde615b37f3fa`.
BERT image: `sha256:e2f307b1a92964d2b076745081af00b96a9a675b50ecef907ed9d321d49fe9b8`.
Both IDs matched after transfer to node 5. BERT pretrained assets come from
`zhihan1996/DNA_bert_6` revision `c56e67ea5827e0ddc67ef059addcf71569b1216e`.
The Docker build now makes saved weights readable by the host user; previously
safetensors mode 0600 caused a misleading missing-file error at runtime.
Asset hashes, source pins and package versions are saved under
`audit/oligo_bert/`. Both containers passed real two-epoch checkpoint-reload smoke
runs with record-ID-preserving prediction export. Node5 detected its RTX A4000.

`run_oligo_bert_matrix.py` runs on node 4 and stores live logs centrally. It seals
input/code/image manifests before launching each isolated run on node 5; returns
all run files with rsync; verifies every SHA256; checks prediction IDs, labels and
finite outputs; appends the central run index; then deletes only the returned run.
Frozen RNA-FM preparation may be reused across seeds of the identical partition:
its source and destination manifests must match after removing only training seed.
The cache is discarded after the third seed. No trained model is reused.
Central index: `evaluation/comparator_runs/oligo_bert_run_index.csv`.
Controller log/PID: `setup/oligo-bert-matrix.log` and `.pid`.
No completed scientific comparison is claimed while these runs are in progress.

First full run: OligoFormer grouped fold 0 / seed 0 completed in 655.8 seconds,
was copied to node 4 with all hashes and prediction checks passing, and its run
directory was removed from node 5. The central index records its artifacts.
BERT-siRNA grouped fold 0 / seed 0 also completed, in 179.3 seconds, and passed
central checksum, record-ID, label and finite-output verification. OligoFormer
grouped fold 0 / seed 1 completed in 567.0 seconds using the verified frozen
preparation cache. The central run index is the authoritative live completion
record; benchmark comparisons await the complete matrix.

The upstream OligoFormer process demo forms auxiliary binary labels with `>0.7`;
the main wrapper uses `>=0.7` on harmonized rounded labels. This does not affect
the approved main R² checkpoint selection. The separately staged original
joint loss/AUC stopping control restores strict `>0.7`. Main runtime code remains frozen.


## Primary sharding and original-schedule queue

After grouped fold0/seed2 finished naturally, the central controller was restarted
with a grouped-only restriction. Only the controller was suspended; its training
child continued to completion. The completed run was checked, returned and indexed
without repeating preparation/training/inference. Its elapsed_seconds field is the
5.9-second return/resume stage, not the original training duration; the full log is
preserved. No numerical runtime source or image changed during this handoff.

Node5 now owns grouped primary runs. Node6 owns random primary runs after AttSiOff finishes. Concurrent execution with
the remaining siRNADiscovery primary runs was approved after a fresh GPU inventory
showed15.8GiB free; our worst preparation stage fits comfortably alongside it. The node6 queue also requires a
verified staging marker. Independent indexes prevent concurrent writer collisions:
`evaluation/comparator_runs/oligo_bert_run_index.csv` (grouped) and
`evaluation/comparator_runs/oligo_bert_random_run_index.csv` (random). The union must
contain exactly sixty unique verified primary run identities. Do not concatenate
an old snapshot of an index with its replacement.

Original-schedule checks are queued on node5 after its thirty grouped primary runs
are verified, while node6 may continue its random primary runs: grouped fold0, seeds0/1/2 for each tool, full1047-row HeLa inference.
They use separate `revision_v1_original` run directories and
`evaluation/comparator_runs/oligo_bert_original_run_index.csv`. Their isolated
runtime snapshot corrects the two legacy original-mode differences described below.

Controller-only orchestration adds axis/fold/index selection, a separate original
series, a stop-before-next-run marker, and a completion-metadata guard. Six focused
tests check empty/new runs, incomplete checkpoints and both methods' main/original
settings. Existing completed checkpoints without matching end-of-training metadata
are rejected rather than silently resumed.

Queue logs and PIDs are `setup/oligo-bert-random-queue.{log,pid}` and
`setup/oligo-bert-original-queue.{log,pid}`. To stop before the next new run, create
`setup/oligo-bert-grouped.stop`, `setup/oligo-bert-random.stop`, or
`setup/oligo-bert-original.stop`. This does not interrupt an active fit. Node6 has an
independent temporary root `/SCRATCH/dtzim01/sirbench-revision-20260921-oligo-bert-node6`.
The runtime snapshot copied from node5 is preserved centrally in
`setup/frozen-oligo-bert-runtime`; all119 numerical source hashes match the original
successful manifests. Per-run return/checksum/ID/label checks and cleanup are unchanged.

## Isolated original-schedule runtime correction

Before any schedule-control training began, the waiting original queue was stopped
and a separate copy was created by `stage_oligo_bert_original.py`. The snapshot is
`runtime_snapshots/oligo-bert-original-v2`, staged temporarily on node5 at
`/SCRATCH/dtzim01/sirbench-revision-20260921-original-node5`. Primary worker trees
were not edited. The complete copied tree passed1172 file checksum checks.

Only BERT checkpoint saving/metadata and Oligo auxiliary-label construction differ
from the frozen runtime: BERT saves the final epoch after all30 epochs; Oligo uses
strict `>0.7` for auxiliary AUC labels. BERT still uses evaluation mode, and does
not reproduce per-epoch test scoring from the demonstration script. The new
controller rejects BERT original-mode artifacts unless metadata confirms30
completed epochs and final-epoch selection at zero-based epoch29.

`audit/oligo_bert/original_schedule_smoke/validation.json` records a real30-epoch
BERT smoke and Oligo preparation at labels0.699/0.700/0.701, which yields binary
labels0/0/1. These fixtures are excluded from every scientific index. Seven
controller checks passed, including rejection of a best-validation checkpoint
presented as a final-epoch result. Source before/after hashes are recorded in
the snapshot's `original_schedule_patch.json`.

The active replacement queue is `setup/oligo-bert-original-queue-v2.{log,pid}`.
It waits for the30 grouped primary runs, then performs the same six planned
schedule controls. Reproduce these controls with the staging script and this
queue; the legacy generic `--original` wrapper alone does not apply these patches.
