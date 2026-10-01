# AttSiOff and siRNADiscovery comparator audit

Status: wrapper corrections and asset recovery are in progress; no final benchmark
results should be inferred from smoke tests. The native siRBench ensemble and
Agentomics remain outside this work.

## Source identity

The AttSiOff paper (doi:10.1007/s44258-024-00019-1; full article at
https://d-nb.info/1335044531/34, availability statement p.11) names
https://github.com/2333liubin/AttSiOff. Pinned commit:
`f1d8ed6fdd93cc2708d910a411c26d3f3fdf08db`.

The siRNADiscovery paper (doi:10.1093/bib/bbae563;
https://pmc.ncbi.nlm.nih.gov/articles/PMC11539000/) names
https://github.com/BertramLoong/siRNADiscovery. Pinned commit:
`ef4741194e6b7773c07cd7ef934b1ee52d9e7d00`.
The alternate GitHub owners in a later review do not supersede these primary
availability statements. An archived Discovery article XML is in
`audit/sirnadiscovery/paper.xml` outside the checkout.

## AttSiOff

| Item | Released implementation | Original siRBench wrapper | Revision action |
|---|---|---|---|
| Adam | lr .005, weight decay .0005 | same | retained |
| Scheduler | cosine warm restarts, T_0=20, T_mult=4 | omitted, even under --original-params | restored |
| Gradient clipping | norm 5 | omitted | restored |
| Training | batch128, at most1000 epochs, patience20, Spearman selection | common100epochs, R2 selection | author-approved common100epochs/patience20/global continuous validation R2; original mode remains explicit |
| PSSM | fitted separately to each evaluation pool | same | training-only PSSM saved with model and used for validation/test; --legacy-pool-pssm restores cohort-specific feature |
| RNA-FM assets | upstream indexed in one source table | every split restarts indices0..N; cache considered complete if any embedding exists | sequence/context hash IDs and complete per-record cache checks prevent cross-split embedding substitution |
| Optional scores | s-Biopredsi, DSIR, i-score loaded but never used in model.forward | zeros where missing | no missing-model-input claim should be made for these three columns |
| Input extent | embedding widths21/59 | centered padding/truncation19/57 to21/59 | retained, explicitly an adaptation |
| Positional encoding | after permutation to[L,B,D], adds encoding over B | same | preserved in faithful primary, defect documented; corrected-position sensitivity must be labeled |

The positional defect is reproduced with the real upstream model in evaluation
mode: reordering four samples changes predictions (maximum absolute difference
0.00721 for untrained seed0), and encoding varies across samples rather than
nucleotides. This establishes input-order dependence, not its effect on trained
benchmark accuracy. Optional score perturbation gives exactly identical predictions.
Evidence: `audit/attsioff/upstream-runtime.txt`; executable check:
`tests/runtime_attsioff_checks.py`.

The fixed RNA-FM preparation is tested with two distinct splits, reused cached
assets, and a missing-embedding failure. Stable record_id values are preserved in
prepared CSVs and prediction exports. Existing released datasets/results remain
unchanged.

## siRNADiscovery

The official siRNA-split configuration uses batch64, lr.001, MSE, HinSAGE
layers[64,32], hops[12,6], dropout.1 and26epochs. The wrapper preserves these model
settings but adapts siRNA length21 to19 and maximum mRNA length9756 to57.
The primary revision uses the author-approved common100epoch ceiling,
patience20 and global validation R2. `--original-params` retains26epochs.

The revision uses training-only graph interactions and isolated validation/test
query graphs, with global accumulated R2 and best-weight restoration even when
the epoch ceiling is reached. Those shared graph changes are implemented and
verified separately by the parent audit.

Thermodynamic sequence features explicitly convert T to U before calling the
upstream function. Missing RNA-AGO2 features are not silently imputed for main runs.

RPISeq feature recovery uses the official public HTTP batch service, with human
AGO2 UniProt Q9UKV8 (protein sequence snapshot saved). Five original public
21-nt guides reproduce their supplied RF probabilities exactly (.75,.65,.70,.80,.65).
The service accepts submitted X-padded contexts; exact requests are retained.
Features for all corrected19-nt guides and57-nt contexts are requested in batches
of at most100. Inputs, raw HTML, request/response hashes, timestamps, endpoint,
protein checksum and output CSV hashes are archived under
`datasets/corrected-v1/RNA_AGO2/`. The retrieval script rejects absent, duplicate,
unexpected or invalid probabilities. HTTP is the documented working endpoint;
HTTPS has a host certificate mismatch. No labels are sent to the service.

The published cofold preprocessing allocates two extra positions then deletes the
first two context positions. Actual ViennaRNA2.4.18 dotplots number bases1..L1+L2
without separator bases; a G9&C9 check confirms this in
`audit/sirnadiscovery/cofold-index/`. The wrapper reproduces this upstream feature
algorithm. Preserve this behavior for faithful primary comparison; any corrected
indexing sensitivity must be explicitly distinguished.

## Verified execution status (2026-09-21)

Both native environments passed real preparation, two training epochs, saved-model
reload and prediction smoke tests. AttSiOff runtime uses PyTorch2.7.1+cu118 and
Ubuntu's ViennaRNA Python binding2.4.17; Discovery uses TensorFlow2.4.1,
StellarGraph1.2.1 and ViennaRNA2.4.18. Node6's RTX A4000 is detected by both.
All8196 AttSiOff RNA-FM arrays have the expected19/57 by640 shapes and finite
values. All4098 rows have finite Discovery self-fold, cofold and AGO2 features.
The input/feature archives and pretrained checkpoint checksums are retained.

Main matrices run on node6 using temporary staging. Each complete run is copied to
node4, verified against SHA256SUMS and checked for matching record IDs, labels and
finite predictions before its temporary run directory is removed. Central indices:
`evaluation/comparator_runs/attsioff_run_index.csv` and
`evaluation/comparator_runs/sirnadiscovery_run_index.csv`. Logs are in `setup/`.
The first returned AttSiOff run took52seconds and the first Discovery run102seconds;
these are timing observations, not final scientific comparisons.

On Discovery grouped fold0, seed0, strict and legacy-transductive inference with
identical weights produce exactly identical predictions for656test and1047HeLa
rows (maximum difference0). This finding applies to the current representation
with unique guide/context hubs. It does not justify transductive inference when
full mRNA creates shared hubs. Evidence:
`audit/sirnadiscovery/graph_sensitivity.json`.

AttSiOff's optional `--correct-position-encoding` training flag uses nucleotide
positions. Its value is saved with the model and restored automatically for test
inference. This is a labeled sensitivity, not the faithful primary architecture.
The corrected runtime check is invariant to sample permutation (maximum difference0).

Discovery's primary configuration is explicitly the published **siRNA_split**
configuration. The repository also contains an mRNA_split configuration with
positional dimension3, hops[4,2], dropout.3 and38epochs (same lr,batch,hiddenlayers).
A completed grouped-fold0 sensitivity distinguishes this full alternative
configuration from the26epoch schedule-only comparison. Main settings stay frozen.

For AttSiOff original-extent sensitivity, the paper defines21-nt antisense input
with its first19nt as the targeting core, and59-nt mRNA as that19-nt site plus20nt
on either side. Experimentally listed 21-mers from the Ichihara supplementary data
provide candidate original inputs when their first 19 nt match a benchmark guide,
but this match also requires source/assay provenance validation before it is
described as recovery of the retained experimental duplex. Overhangs must not
be invented from a target transcript. Original and standardized
inputs must be evaluated on the same eligible records and fixed partition.

### Completed primary matrix and isolated sensitivity preparation

AttSiOff completed all 30 primary runs (five grouped folds and five unselected random partitions, three seeds each). All predictions, checkpoints, training metadata and run manifests were copied to node4 and checked against remote SHA256 lists before the corresponding temporary node6 run directories were deleted. The central index is `evaluation/comparator_runs/attsioff_run_index.csv`; the independent matrix-completeness/metadata/prediction check is `audit/attsioff/main_matrix_validation.json`. Every primary run uses the upstream batch-position encoding, training-only PSSM and the agreed common validation selection policy.

The isolated sensitivity worker/controller in `benchmark/revision/run_node6_sensitivity.py` and `control_node6_sensitivities.py` use explicit JSON specifications and the frozen wrappers without modifying their source. Completed jobs may be returned again only when their specification and completion checksums match. Partial model directories fail closed. Tests cover rejection of a sequence/feature mismatch and corrupt completed output. The primary run controllers similarly reject checkpoints lacking training completion metadata on any attempted restart; the already-running fresh matrices were checked separately.

`discovery_structure_cache.py` preserves the exact frozen preprocessing functions, including fixed-size zero padding, the upstream two-position cofold deletion and seeded truncated SVD. Independent sequence/pair tasks have content hashes covering sequence, geometry, source checksum and software versions. Each cache entry retains the compressed ViennaRNA dotplot, vector, completion metadata and checksums. A completed entry is reused only after checksum/shape/finite-value validation. The real Discovery image regression compares serial and parallel outputs exactly, verifies resume leaves completed cache entries untouched and rejects corruption (`audit/sirnadiscovery/structure-cache-runtime.txt`). Workers use one BLAS/OpenMP thread each. The runtime is the same ViennaRNA 2.4.18 Discovery image as the primary benchmark; package versions are embedded in every input manifest and cache key.

The frozen paired archived-target cohort has 3730 eligible interactions with 83 distinct target sequences (median 720 nt, maximum 9259 nt). These are archived source target sequences; some references are short fragments, so this sensitivity does not claim recovery of every original full assay transcript. The isolated full-length pipeline uses the original 9756-position matrix extent without truncating eligible targets. A four-worker pilot completed; the full cache then launched with 16 bounded CPU workers on temporary node6 storage while model training continued. After confirming spare capacity, the cache was restarted with 32 and then 48 workers; completed entries were checksum-verified and reused, and incomplete tasks were recomputed. Each worker remains limited to one BLAS/OpenMP thread, leaving capacity for the independent 32-worker ENsiRNA preparation and model training. Logs from earlier worker counts are preserved. The numerical source and model jobs were unchanged. The AGO2/RPISeq archive for this cohort contains exact public sequence requests and returned probabilities for 3730 guides and 83 targets. Its preparation remains separate from the 57-nt primary assets.

The Discovery `mRNA_split` sensitivity changes the published model/training configuration (dmodel 3, hops 4/2, dropout 0.3, 38 fixed epochs) while retaining standardized 19/57 inputs. This is explicitly distinct from the paired archived-target extent experiment and from the 26-epoch schedule-only comparison. The 38-epoch experiment is not described as exact reproduction of the native full pipeline. Both fixed-epoch comparisons use the final checkpoint, whereas the paired input-extent experiments use the same common 100-epoch/patience-20 validation-R² selection policy as the main benchmark.

The 2596 candidate AttSiOff 21/59 assets were extracted separately and checked for correct array dimensions and finite values. A unique matching 19-nt core alone does not establish that an experimentally listed 21-mer belongs to the same retained assay record. The subsequent source/scaled-label audit retained 2507 compatible records and excluded 89 unresolved candidates. All six paired runs on this narrower v2 cohort are returned and verified: 1506 training, 441 validation, 517 test and 43 eligible HeLa records per variant. The existing sequence-specific embeddings were reused, and each training PSSM was fitted to the new training subset. These are source-compatible reconstructed 21/59 inputs; their 43-row HeLa subset must be distinguished from the primary 1047-row transfer set (`audit/attsioff/extent_v2_validation.json`).

A trained AttSiOff order check used the same grouped-fold-0/seed-0 weights and all 656 test records, with identical CPU inference settings and batch size 128. Reversing the record order changed individual predictions by a maximum of 0.08126 and a mean of 0.00951 after restoring record IDs; Pearson correlation changed from 0.60802 to 0.61062. This demonstrates order dependence for this checkpoint, without establishing its effect on all rankings (`audit/attsioff/trained_order_sensitivity/summary.json`). The three corrected-position training runs are separately indexed and verified on node4.

All 3730 archived-target sensitivity records have exactly one occurrence of the complementary 19-nt guide site, so the upstream first-match position lookup is unambiguous for this cohort (`audit/sirnadiscovery/full_rna_binding_sites.csv`). The recovered RPISeq guide probabilities exactly match the primary archive for all 3730 unchanged guides; 83 context probabilities were retrieved from the archived target sequences and their output hashes verified (`audit/sirnadiscovery/full_rna_rpiseq_validation.json`).

### Released AttSiOff stopping schedule

The released AttSiOff trainer caps training at 1000 epochs and monitors validation Spearman correlation rounded to three decimal places. Its `EarlyStopping` class uses patience 20, and a tie replaces the checkpoint and resets patience. Consequently, the existing wrapper's `--original-params` option alone does not reproduce the released stopping semantics: it used unrounded correlation and strict improvement.

The separate `attsioff_original_schedule.py` entry point changes only this selection rule in an integrity-checked copy of the frozen trainer loaded in memory. The primary source file is not edited. Both the entry point and effective training source are saved with each sensitivity run. Tests compare checkpoint selection and stopping directly against the released `EarlyStopping` class, including rounded ties, 1000 consecutive tied scores and undefined scores. The sensitivity retains primary 19/57 inputs, the faithful upstream architecture and training-only PSSM, and uses the released 1000-epoch cap without shortening it. Seed 0 is profiled first; seeds 1 and 2 are specified separately in the central matrix.

Both 30-run primary matrices are complete and verified on node4. The source snapshot at `runtime_snapshots/node6-attdisc-v1` reproduces all 123 source-file hashes recorded in the primary AttSiOff and Discovery manifests. The original-extent Discovery feature computation and all three paired training runs are complete. The full feature archive and all run artifacts were returned to node4 and verified against their SHA inventories.

The first released AttSiOff schedule run (fold 0, seed 0) completed in 61.6 seconds including verification/return. It retained the 1000-epoch cap and stopped after 26 epochs; the checkpoint selected at epoch index 5 had rounded validation Spearman 0.585. This is a runtime observation for the prespecified first seed, not justification for reducing the cap on subsequent runs.

All three AttSiOff released-stopping runs are now verified and returned. Their cap, patience, rounded-Spearman/tie semantics, unchanged input contract, training-only PSSM, archived entry points and effective trainer were checked in `audit/attsioff/original_schedule_validation.json`. They remain sensitivity runs in `evaluation/comparator_runs/node6_sensitivity_run_index.csv`; they do not replace any primary model or primary metric.

The five original 21-nt Discovery calibration guides also reproduce the released RNAfold/SVD feature vectors within CSV precision (maximum absolute difference below 3.5 × 10⁻¹⁰; `audit/sirnadiscovery/fold-calibration/summary.json`). This verifies those five reference vectors, not the entirety of the original experimental pipeline.


### Final sensitivity execution and cohort accounting

All 24 specified sensitivity runs are complete and verified on node4: 12 AttSiOff and 12 Discovery runs across eight cases, each using the prespecified grouped fold 0 and seeds 0, 1 and 2. The two primary matrices remain separate, with 30 runs per method across five grouped folds and five random partitions. `audit/attdisc-sensitivities/cases.csv` records the actual input counts, completed seeds and training policy for each case; `runs.json` retains the effective training metadata. Input-file hashes, ordered record IDs, seed identities and sensitivity completion seals were checked again when this summary was generated.

The paired Discovery extent experiment uses 1814 training, 379 validation, 626 test and 911 eligible HeLa records per variant. Both variants use the common 100-epoch/patience-20 validation-R² policy and strict graph protocol. All three paired seeds have identical evaluation IDs and labels (`audit/sirnadiscovery/extent_validation.json`). These results concern an eligible archived-target cohort; its 911-row HeLa subset is not the primary 1047-row transfer set. The standardized and reconstructed AttSiOff comparison similarly uses its explicitly labeled 43-row eligible HeLa subset.

The completed Discovery feature archive contains 3730 guide vectors, 83 target vectors and 3730 interaction vectors; dimensions, identifiers, finite values and all transferred file hashes were verified (`audit/sirnadiscovery/full_rna_asset_validation.json`). Full-length cofolds had a substantial runtime tail. Increasing only the bounded worker count preserved the frozen numerical functions and checked completed cache entries before reuse. No target was shortened to accelerate computation. All predictions and feature artifacts remain in the node4 workspace.
