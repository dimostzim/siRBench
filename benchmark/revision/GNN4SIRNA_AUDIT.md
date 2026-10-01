# GNN4siRNA revision implementation audit

This audit concerns retraining the released predictor under the revised benchmark.
It does not establish equivalence to every result in the original publication.
The pinned upstream commit is `5247663c6eb3a4939f1eb7f385f9be91d6324d60`.
Paths below are relative to `/SCRATCH/dtzim01/sirbench-revision-20260921`.

## Architecture and training

The wrapper retains HinSAGE layers32/16, sampled neighbors8/4, dropout0.15,
batch size60, MSE loss and learning rate0.001. The optimizer is Adamax, as in the
publication's Table1; the released executable instead instantiates Adam. This
paper/repository discrepancy must remain explicit in the comparison table.
The common policy uses up to100 epochs, patience20 on complete-validation-set R²,
and restores the best checkpoint even when training reaches the epoch cap.
The original-schedule sensitivity uses10 epochs without early stopping and the
same paper-specified optimizer. It is a stopping-schedule sensitivity, not an
exact reproduction of the released executable's optimizer choice.

The runtime uses TensorFlow2.4.1, StellarGraph1.2.1 and ViennaRNA2.4.18 in a pinned
Docker image. Each run records configuration, seed, completed epochs, validation
history, selected checkpoint information and graph protocol. Inputs and results
have stable record IDs; controller validation checks labels, completeness and
finite predictions before marking a run verified.

## Preprocessing corrections and preserved conventions

The original wrapper wrote guide FASTA using T although the upstream
thermodynamic table checks RNA bases. Guides are now written using U; upstream
k-mer extraction performs its own U-to-T conversion. Stable benchmark record IDs
replace positional row identifiers when supplied.

The main target is the common57-nt context. Upstream RNAup preprocessing uses
the reverse complement of the guide; that convention is preserved for this
published-model comparison and distinguished from the shared100-feature table.
Feature generation uses only the supplied sequences and labels are attached by
stable ID. No validation/test-fitted normalization is introduced.

## Inductive graph protocol

Training excludes validation interaction nodes and their edges. A sequence hub
may remain only when attached to a training interaction. Each validation/test
query receives private guide and target nodes, yielding disconnected three-node
components. Thus no query can obtain features from another held-out interaction
or from training interactions at inference. Batching these components is
equivalent to independent per-query inference.

`audit/graph-runtime-checks.txt` records six passing checks: removal of validation
neighbors, private query nodes, invariance to other evaluation samples, complete
validation R² across unequal/constant batches, metric reporting and restore-best
behavior at the epoch cap. These tests ran inside the actual TensorFlow image.

For grouped fold0/seed0, strict and legacy joint-graph inference with the same
strict-trained weights are exactly equal on656 test records and1047 HeLa records
when both are run on CPU in the same image and threading configuration
(`audit/gnn4sirna/graph_sensitivity.json`). This is a limited control on the57-nt
representation, not evidence that transductive access is harmless generally.
An earlier CPU-versus-GPU comparison is preserved separately as
`graph_sensitivity_cross_device.json`; its small numerical differences cannot be
attributed to graph construction.

## Input extent and execution

The paired source-extent cohort contains1814 training,379 validation,626 test
and942 HeLa records in grouped fold0, with three training seeds and identical
IDs/labels in both variants. Archived source targets with exact context matches
are preferred according to the author-approved reconstruction rule. Some source
targets are fragments; the sensitivity restores available source extent and
does not guarantee the complete historical assay transcript.

Long-target RNAup computation uses a resumable cache, with the executable hash,
exact input and arguments in each cache key. Temporary working directories are
separate for each query. Raw stdout/stderr and parsed values are preserved;
corrupt entries fail validation. `audit/gnn4sirna/full-cache-check.txt` verifies
exact agreement with the published serial preprocessing on three real rows,
unchanged cache entries on resume and rejection of corrupted entries.
`subset_gnn_features.py` matches canonical features by ID and exact sequence,
then attaches current labels; the fold0 training subset reproduces direct
preparation exactly. Active primary numerical code was not replaced by the
parallel preprocessing helper.

Primary results: `siRBench/benchmark/competitors/runs/main-gnn-v1/run_index.csv`.
Separate sensitivity results use `gnn-original-schedule-v1` and
`gnn-full-extent-v1` under the same runs directory. A completed experiment is
identified by a verified index row and validated artifacts, not this audit text.
