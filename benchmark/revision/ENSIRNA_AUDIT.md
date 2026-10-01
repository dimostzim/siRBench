# ENsiRNA audit for the revision

Status: wrapper corrections, CPU regressions, real PyTorch checks, full training/reload
smoke, and canonical-cache identity/label-injection checks pass. Canonical structures
are being prepared with the authors' exact Rosetta runtime. No revised ENsiRNA
scientific results are available yet.

## Source identity and configuration

The [2025 Journal of Molecular Biology paper](https://doi.org/10.1016/j.jmb.2025.169131)
links `tanwenchong/ENsiRNA` in its data-availability statement. The audit pins
[`028824341635903f3c661f5d1cc737de106493d5`](https://github.com/tanwenchong/ENsiRNA/tree/028824341635903f3c661f5d1cc737de106493d5),
the same commit cited by Reviewer 1. The official training instructions use
`train.sh config.json`; CLI defaults alone are therefore not the released configuration.

| Setting | Official configuration / implementation | Previous wrapper behavior | Revised behavior |
|---|---|---|---|
| Initial/final learning rates | 1e-4 / 1e-5 | Standard command correct; `--original-params` used 1e-3 / 1e-4 | Both configuration paths use official rates |
| Maximum epochs | 100 | Standard command 100; original path 10 | 100 |
| Embedding / hidden size / layers | 128 / 256 / 2 | 128 / CLI default 128 / CLI default 3 | 128 / 256 / 2 |
| Neighbors | 9 | 9 | 9 |
| Training shuffle | true | Not forwarded; CLI default false | true |
| Optimizer | Adam | Adam | Adam |
| Learning-rate scheduler | ReduceLROnPlateau, factor 0.9, patience 20, minimum 1e-5 | `mode=min` even when selecting by increasing R² | Direction follows selected metric |
| Model selection | Original validation loss | Standard wrapper adds global R²; test selected minimum score despite R² maximization | Test takes first trainer-ranked checkpoint; no implicit ensemble across runs |
| Early stopping | Patience compares with previous epoch | Same previous-epoch comparison | Primary compares with global best; `--legacy_stopping` preserves original semantics for sensitivity |

The five released checkpoints load successfully and each contains 3,267,812
parameters; their graph layers confirm 128 input dimensions, 256 hidden dimensions,
and two message-passing layers. The deployment `design.sh` ensembles those five
pretrained checkpoints. Primary revised retraining instead evaluates one
validation-selected checkpoint per training seed under the common protocol; any
released-checkpoint reproduction is reported separately from benchmark results.

The official `final_lr` is used to calculate an exponential-decay parameter, but
that exponential scheduler is commented out upstream. The active plateau
scheduler has a minimum learning rate of 1e-5. The revision retains the active
released scheduler.

The historical EN run is now partially established by its `namespace.json` in
[Zenodo record 20070541](https://zenodo.org/records/20070541), rather than by a
`train_meta.json`. It records lr=1e-4, final_lr=1e-5, max_epoch=100, batch_size=16,
patience=20, and validation R²: the archived run therefore used the released
learning rates, despite the separate `--original-params` branch defect identified
by the reviewer. It also records hidden_size=128, n_layers=3, and shuffle=false,
confirming the architecture and shuffle mismatches in the actual historical run.
The archived TensorBoard stream contains 15,600 training-loss steps (0–15,599),
with 156 steps per epoch in the namespace, confirming that all 100 epochs ran.
The training-loss timestamps span 88.8 minutes; historical hardware is unrecorded.
Its archived test/transfer Pearson values (0.33672544798103066 and
0.4280343253844821) match the manuscript values. Extracted namespace, metrics,
and their SHA256 values are retained in `audit/ensirna-archived-*`.

The archived `topk_map.txt` ranks epoch19 highest (validation R² 0.17784804898758844)
and epoch8 lowest among retained checkpoints (0.12331253758387584). Both prediction
workbooks contain only `result_0`, without a checkpoint identifier. Therefore the
old minimum-score selection code is a confirmed defect, but the archive does not
independently identify the checkpoint used for those predictions. The namespace
also does not record a training seed. These limits should remain explicit.

## Representation and preparation

Official `train_1.csv` and `valid_1.csv` contain 2,252 and 564 rows, respectively;
both strands are 19 nt. The original pipeline receives full mRNA plus a zero-based
site position and crops a 61 nt context internally. Official `test.csv` contains
702 correctly oriented Takayuki records against the 720 nt EGFP reference.

The harmonized 57 nt contract is retained for the main comparison, with explicit
padding to the network's 61 nt representation. Original-extent sensitivity requires
61 nt contexts, rather than changing the network to process an entire transcript.
An audit of all 4,098 primary contexts found the intended sense target at the
center, with no disagreement between that position and the first-match lookup.
Extra flanks must be supported by the recovered references and agreement among
candidate transcripts; uncertain targets must be reported rather than invented.

The previous siRBench patch substituted an ideal perfectly paired 19 bp duplex for
upstream RNAplex secondary structure before Rosetta folding. An audit
with the pinned ViennaRNA 2.6.4 runtime found different structures for 525 of 4,098
corrected records. All 4,098 RNAplex-derived position/chain vectors agree with
the adapter position indexing; no row fails position-metadata construction.
Primary revised preparation restores RNAplex-derived pairing and runs Rosetta
with an explicit fixed preprocessing seed. The old ideal-duplex adaptation is
not silently retained as the published architecture.

RNA-FM embeddings contain BOS followed by sequence tokens. Upstream left-padding
was prepended before BOS, moving that embedding away from the graph's global
mRNA node. With 57 nt inputs this affects every record. The revised adapter inserts
padding after BOS and before any antisense suffix; a sentinel-tensor check covers
both positions. This is an explicit padding correction, not a learned-model change.

Other preparation corrections:

- Keep internal unknown context bases in place; trim only terminal padding.
- Require the sense target at the supplied/inferred position, rather than falling
  back to antisense matching or position zero.
- Preserve `record_id`; reject duplicate/missing IDs.
- Existing PDBs no longer spuriously require Rosetta (`~bool` was always truthy).
- Fail on missing rows, failed Rosetta commands, invalid PDBs, or failed PDB parsing.
  A Rosetta silent file is never substituted for a PDB.
- Write completed PDBs and JSONL atomically; bound Rosetta workers to four by default.

## Reuse, reproducibility, and verification

Canonical preparation is per-record: fixed pretrained RNA-FM, ViennaRNA features,
and Rosetta structures; no training labels or cross-record feature fitting enter
these transformations. Every processed feature carries its explicit record ID.
`subset_features.py` freezes hashes of canonical records/parts, source code, PDBs,
and supplied assets, validates them before reuse, verifies sequence/context for
each requested ID, and injects labels from the requested split by ID. It does not
reuse labels or feature rows by positional ordering.

The ENsiRNA image builds ViennaRNA 2.6.4 CLI from the official source archive and
pins its Python bindings to 2.6.4. The source archive SHA256 is
`3a997a6aa6a3ce1af4898aa559acb053e820aa74bac06ef7726b9aa97a053788`.
All assets and run artifacts remain under the node4 SCRATCH revision workspace.

CPU regression suite: 19 checks pass (configuration, preparation,
checkpoint selection, subset identity/label injection, bounded scheduling,
drain behavior, finite predictions, final training metadata, and rejection of
unverified setup assets without replacing existing runtimes). Runtime checks in
`tests/runtime_ensirna_checks.py` pass with actual PyTorch, covering BOS padding,
scheduler direction, global-best stopping, failure atomicity, and a numerical
global-R² check with unequal validation batch sizes. RNA-FM
(99,521,546 parameters) loads on the node4 NVIDIA A10. The 16-record end-to-end smoke (8 training, 4 validation, 4 test) generated
real PDBs, trained the released architecture for two epochs, reloaded its selected
checkpoint, and produced four finite, ID-matched predictions. Training plus feature
preparation took 10.98 s and reload/prediction 7.57 s; these tiny-cohort timings and
metrics are smoke checks, not benchmark results. A further two-record check with
release-371 passed structure parsing and feature extraction. Real cached features
were then reordered with deliberately changed probe labels, verifying label
injection by ID without modifying canonical features.

Original-extent eligibility is frozen at `datasets/input-extents-v1/`: 3,920 rows
have conservative 61 nt contexts, while 171 conflicting extensions and seven
site-only mappings are excluded. The planned matched comparison uses eligible
records from grouped fold0, seeds0–2, with both 57 nt and 61 nt inputs. The 61 nt
cache reuses the same duplex PDBs and recomputes only context-dependent features.

## Rosetta asset provenance and distributed preparation

The wrapper's historical URL contains `3.15`, but its downloaded bundle actually
reports release-408 / `2025.37+release.df75a9c48e`. It was used only for the initial
functional smoke and is not silently described as Rosetta 3.15. The public PDB link
redirected to Google sign-in during this audit; no coordinates were obtained there.

The authors' `tanwenchong/ensirna:v2` Docker application layer contains Rosetta
release-371, version `2024.09+release.06b3cf8`, commit
`06b3cf8ad0940d628690d0ed6fa2009d72ad2b44`. It contains no ENsiRNA example PDBs;
the PDB files in that layer are Rosetta database/test assets. The layer SHA256 is
`376d3c0a0fd79225ec273944bee58e3711e346ba61f0293f415410c806e2323c`.
Primary canonical folding uses this exact released runtime and its database,
RNAplex 2.6.4 secondary structures, the unmodified 36,000-cycle folding settings,
and explicit `-constant_seed -jran 0`. Runtime-file hashes and the original image
manifest are retained under `audit/` and `setup/ensirna-shards/`.

The 4,098 records are deterministically sharded by row index: 2,050 on node4
(32 workers), 1,024 on node5, and 1,024 on node6. The auxiliary nodes began
with 16 workers and increased to 32 after capacity checks; completed PDBs were
parsed, checked against both strand sequences, and SHA-verified before reuse. Sharding affects only
independent structure calculation, not benchmark partitions or labels. Nodes5/6
hold temporary preparation directories; PDBs and logs are copied to node4 and
hash-verified before their worker directories are removed. Canonical features,
training runs, and permanent evidence remain in the node4 revision workspace.

## Planned stopping-policy contrast

A separate sensitivity uses the complete primary grouped fold0 with training
seeds0–2 and full HeLa transfer. It retains the same corrected records, features,
architecture, optimizer, learning rates, and 100-epoch limit as the primary runs.
The original mode selects minimum validation loss, uses the upstream patience
of 1,000 with previous-epoch comparison, and runs the active plateau scheduler
in minimization mode. The primary mode selects global validation R² and stops
after 20 epochs without improvement over the global best. This is a real stopping
and selection contrast, separate from the paired 57/61 nt input comparison.


## Loader throughput and concurrent execution

The revised training loader uses four CPU workers and prefetch factor one. The
cached dataset has no stochastic augmentation. A fixed-seed 64-step check found
identical input tensor hashes and sample order for zero versus four workers;
maximum loss differences were below 4e-8. A separate 64-step check retained the
upstream anomaly-detection setting and compared one versus two independent GPU
training processes. Both concurrent processes retained identical tensor hashes
and sample order, with maximum loss differences below 8e-8 relative to the solo
run. Each used about 1.04 GiB of GPU memory. On this bounded repeated-smoke-data
profile, two processes completed in 15.71 seconds, compared with twice the solo
9.24-second interval. These figures assess runtime feasibility only and do not
predict scientific training duration or performance.

The main launcher therefore allows two independently seeded runs concurrently
on node4. No mixed precision, architecture change, altered numerical settings,
or reduced training budget is introduced. The drain marker at
`setup/ensirna-stop-scheduling` stops new scheduling while allowing active runs
to finish. Completed runs are admitted to the index only after checking ordered
record IDs, labels, finite predictions, the full resolved training configuration,
completed epoch/step counts, and agreement between final metadata and the
trainer-ranked checkpoint. Each index row includes the actual
`models/ensirna/version_0/training_metadata.json` path. Profiling inputs, hashes,
logs, and CUDA observations are retained under `audit/ensirna-*-probe*`.


The final isolated matrix smoke exercised the real shared runner, canonical subset
extraction, two concurrent jobs, checkpoint reload, both prediction outputs, and
final metadata validation for seeds 0, 1, and 2. They stopped after 28, 26, and 27
epochs, respectively, exactly 20 unimproved epochs after their selected best
checkpoints. All three index entries passed identity, label, prediction-finiteness,
and resolved-policy validation. The four-row smoke transfer input deliberately
reuses smoke test records and is not a scientific transfer result. Evidence is
`audit/ensirna-matrix-smoke-check.json`; no smoke row enters the primary index.


## Pinned public setup

`fetch_rosetta.sh` now retrieves the same operational subset from the immutable
image manifest SHA256
`1a9c8b80a2d5b5943770fb5e736264cb5234997181df98d7704bde673f09167f`.
The raw manifest is included as text; binaries are retrieved from the authors'
registry and are not added to the code archive. Installation verifies the image
manifest, application-layer digest, every operational file via the aggregate
SHA256 manifest (`a2008e7d09e25bf5d93d4432c826bf5723aecb9341556edba260503b9700217d`),
all symbolic-link targets, and both executable digests. The accompanying license
and actual version/provenance are saved beside a fresh installation. Existing
unmatched runtimes fail validation and are not overwritten. There is no mutable
URL fallback; source cloning also pins the audited ENsiRNA commit.

Fresh cached-layer installation, repeat verification, and verification of the
actual primary runtime all passed. Comparing the full raw package with the
operational subset found that every staged operational file matched its raw counterpart;
73,019 unused source/build/test/documentation files are omitted from the selected
subset. The installer reproduces that subset exactly. Logs and comparison
manifests are retained under `setup/ensirna-rosetta-*` and
`audit/ensirna-runtime-raw-*`.
