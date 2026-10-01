# Reproducing the revised siRBench benchmark

The immutable experimental snapshot is
[Zenodo version 2](https://doi.org/10.5281/zenodo.23001225). This GitHub update
publishes its revised source and active Agentomics models, adds portable dataset
and archive commands, and pins competitor installation to the observed upstream
commits and Python package versions. It does not change model architectures,
learned weights, frozen predictions, labels or split memberships.

Use separate environments. Competitors retain their tested Docker runtimes.
Agentomics pipelines and TabPFN each have independent locked uv projects.
The root uv project runs dataset preparation and CPU checks. `reproduce/` retains
the exact analysis dependency lock used for archive replay.

## Get the archived inputs

```bash
uv sync --locked
uv run --locked python scripts/fetch_archive.py
export PACKAGE="$PWD/artifacts/siRBench-v2-2026-09-27"
```

Alternatively pass `--zip /path/to/siRBench-v2-2026-09-27-zenodo.zip`.
The command checks the ZIP SHA256 and every file listed in `SHA256SUMS`.
It refuses modified existing files. Allow several GB for extracted inputs and
the separate replay workspace. Generated files are ignored by Git.

## Dataset and partitions

```bash
uv run --locked python scripts/reproduce_dataset.py \
  --package "$PACKAGE" --output outputs/dataset
```

This starts from the released harmonized CSVs in `data/`, rebuilds stable IDs,
applies the archived 702 Takayuki strand corrections without changing labels,
and reconstructs all grouped/random partitions from frozen target mappings.
It verifies the corrected record files and all 36 protocol CSVs that do not
contain absolute paths byte for byte. The 30-row run matrix is checked after
normalizing its filesystem paths. A machine-readable report is written to
`outputs/dataset/dataset-verification.json`.

The default copies the verified 100-feature matrix. To recompute it:

```bash
# RNAfold, RNAcofold and RNAup 2.4.18 must be on PATH.
uv run --locked python scripts/reproduce_dataset.py \
  --package "$PACKAGE" --output outputs/dataset-regenerated \
  --features regenerate --workers 8
```

The command rejects another ViennaRNA version and checks the regenerated matrix
against the archived SHA256. The Python ViennaRNA bindings used by individual
Agentomics models are separate dependencies and do not replace these executables.

The reconstruction starts at the released harmonized data. The original complete
raw-publication curation script is unavailable, so this is not a claim to
recreate every historical exclusion or conflicting-label decision. Archived
`workspace/audit/provenance/` retains recovered source rows, conflict evidence
and remaining uncertainty. Target mapping evidence and reference sequences are
under `workspace/audit/targets/`. New GenBank responses may differ from those
frozen references and must not silently redefine the published groups.

For an independent check against separately acquired, pinned OligoFormer data,
the original correction CLI is also retained:

```bash
uv run --locked python benchmark/revision/correct_takayuki.py \
  --records outputs/dataset/audit/records.csv \
  --upstream /path/to/pinned/OligoFormer \
  --egfp-fasta /path/to/pinned/GNN4siRNA/data/raw/dataset_2/mRNA_2.fas \
  --output-dir outputs/upstream-correction-check
```

The expected upstream input hashes are in the archived
`workspace/datasets/corrected-v1/correction_manifest.json`.

## Numerical results and baselines

Use Linux x86-64 for exact archived-table verification. macOS may differ at
floating-point round-off despite identical dependency versions. The strict
replay deliberately reports such a difference instead of declaring exact parity.

```bash
uv run --frozen --project reproduce python reproduce/replay_results.py \
  --package "$PACKAGE" --output outputs/result-replay
```

Replay uses an isolated copy of the archived source and saved predictions. It
recalculates pooled metrics, per-source/cell-line results, paired contrasts,
split/seed variation, 2,000 group-bootstrap draws, the 12 paired sensitivities
and all five Agentomics results. Its verification report compares every output
table with the archive, allowing only explicitly checked path relocation and
the archived CSV serialization tolerance. It performs no fitting or inference.

To refit the simple baselines on the rebuilt partitions:

```bash
uv run --locked python benchmark/revision/baselines.py \
  --protocol outputs/dataset/evaluation/protocol-v1 \
  --feature-manifest outputs/dataset/datasets/corrected-v1/records_features.manifest.json \
  --output outputs/baseline-refits
```

The seven references are training mean, three ridge representations, calibrated
Ui-Tei functionality, and fixed/calibrated i-Score. i-Score's historical Huesken
training overlap prevents interpreting it as an independent primary comparator.

## Six published predictors

Use Linux x86-64 with Docker, NVIDIA Container Toolkit and a compatible GPU.
TensorFlow/StellarGraph and PyTorch tools retain separate images. The audited
source commits are recorded in `provenance/upstream-source-pins.json`.
Each Dockerfile uses `requirements.frozen.txt` as a pip constraint, taken from
the actual revision image. The original immutable local image IDs are recorded
in `provenance/training-dependency-lock.json`. OS/base-image rebuilds can still
produce different image IDs; the package pins do not claim byte-identical layers.

Build only the desired tool first. The setup script obtains upstream code and
external dependencies; read that tool's README and the third-party notices.

```bash
tool=gnn4sirna
IMAGE_TAG="$tool:revision" bash "benchmark/competitors/tools/$tool/setup.sh" --docker
export SIRBENCH_IMAGE_TAG=revision
```

Valid tool names are `oligoformer`, `gnn4sirna`, `sirnadiscovery`, `attsioff`,
`sirnabert` and `ensirna`. Setup refuses an existing checkout at a different
upstream commit rather than resetting it. RNA-FM weights are downloaded by
the relevant setup scripts. ENsiRNA additionally needs the documented Rosetta
runtime and ViennaRNA 2.6.4. Neither is substituted with the dataset's 2.4.18
feature generator.

Run one primary fit from the repository root:

```bash
export PROTOCOL="$PWD/outputs/dataset/evaluation/protocol-v1"
uv run --locked bash benchmark/competitors/run_tool.sh \
  --tool gnn4sirna --seed 0 \
  --train "$PROTOCOL/grouped/fold_0/train.csv" \
  --val "$PROTOCOL/grouped/fold_0/val.csv" \
  --test "$PROTOCOL/grouped/fold_0/test.csv" \
  --leftout "$PROTOCOL/hela_full.csv" \
  --run-dir "$PWD/benchmark/competitors/runs/reproduction/gnn4sirna/grouped_0_seed_0"
```

Use a **new run directory under the repository** for each tool/axis/fold/seed.
The shell entry point seals input copies, settings, source hashes and the actual
Docker image ID. The complete primary matrix is six tools × two axes
(`grouped`, `random`) × five folds × three seeds (`0,1,2`) = 180 fits.
Primary runs use the common 100-epoch cap and validation-R² stopping policy.
The original schedule and input-extent sensitivities have separate audited
entry points in `benchmark/revision/`; do not add `--original` to primary runs.

For siRNADiscovery, install the archived RPISeq features before invoking the
wrapper. The fallback location below is recognized by `run_tool.sh`:

```bash
mkdir -p data/sirnadiscovery
cp -R "$PACKAGE/workspace/datasets/corrected-v1/RNA_AGO2" data/sirnadiscovery/
```

These are frozen external-service outputs. The retrieval code and request audit
are retained; querying a changing remote service is not assumed to reproduce
the same features. ENsiRNA's more expensive structural features can be reused
through its `subset_features.py` and `run_matrix.py` when a separately verified
canonical feature cache is available.

The primary graph wrappers exclude validation interactions from training
neighborhoods and perform isolated-query inference. OligoFormer's upstream
batch-coupled LSTM behavior is preserved. Prediction order and batch size matter;
changing them is not a harmless inference optimization. AttSiOff's upstream
batch-position behavior is also preserved in primary runs. The respective
audits distinguish these behaviors from labeled sensitivity controls.

The archive includes prediction evidence for all 228 primary/sensitivity fits.
It does not contain the competitors' fitted checkpoints or restricted upstream
source. Those omissions are inventoried, not replaced with dummy models.

## Agentomics reference models

```bash
git lfs install
git lfs pull
export REFERENCES="$PWD/benchmark/revision/reference_models"
uv sync --frozen --project "$REFERENCES/fold_0"
uv run --frozen --project "$REFERENCES/fold_0" python "$REFERENCES/fold_0/inference.py" \
  --input "$PACKAGE/reference-evaluation/fold_0/inputs/test" \
  --output "$PWD/outputs/fold_0_predictions.csv"
```

Repeat with folds 1–4. All five active fitted models are included, with source
and artifact checksums. Each input directory contains `data.csv` with
`id,siRNA,extended_mRNA`. There are no label or assay-covariate inputs.

Verify all five models against validation, grouped-test and full-HeLa predictions:

```bash
uv run --locked python "$REFERENCES/verify_package.py" \
  --evaluation-root "$PACKAGE/reference-evaluation" \
  --output-dir outputs/reference-verification
```

This checks 15 inference runs, row identities, finite predictions and maximum
absolute differences of at most 1e-6. Linux x86-64 is the tested runtime.

To refit a selected pipeline, the dataset command exports the required
`input/data.csv` and `labels.csv` layout:

```bash
uv run --frozen --project "$REFERENCES/fold_0" python "$REFERENCES/fold_0/train.py" \
  --train-data "$PWD/outputs/dataset/reference-training/fold_0/train" \
  --validation-data "$PWD/outputs/dataset/reference-training/fold_0/validation" \
  --artifacts-dir "$PWD/outputs/refit-fold-0/training_artifacts"
```

Use a new output directory. Repeat with the matching pipeline and fold, never
selecting a pipeline on another fold's test labels. These scripts refit the
selected iteration, including its internal candidate selection. They do not
replay the 95 LLM-driven search iterations. Saved-model inference and selected
pipeline refitting require no Agentomics/LLM credentials. See the model README
for completed search budgets and the single-fit-per-fold limitation.

## TabPFN models

Both frozen and fine-tuned variants share
`benchmark/revision/tabpfn/pyproject.toml` and `uv.lock`. Use the detailed
[uv guide](benchmark/revision/tabpfn/UV_REPRODUCE.md) for installation,
training, saved-state inference and model access. Use the archived workspace
as the data root so the original protocol manifest hash remains intact:

```bash
export R="$PACKAGE/workspace"
uv sync --locked --project benchmark/revision/tabpfn
# Obtain the authorized foundation checkpoint at the documented path under $R.
uv run --locked --project benchmark/revision/tabpfn \
  python benchmark/revision/tabpfn/run_benchmark.py \
  --root "$R" --axis grouped --fold 0 --seed 0 \
  --output "$PWD/outputs/tabpfn-grouped-0-seed-0"
```

This creates both variants. Repeating two axes × five folds × three seeds gives
30 frozen context fits and 30 gradient fine-tuning fits. Foundation and fitted
weights are external dependencies, not Git LFS files in this repository. Existing
saved fits can be reloaded with `predict_saved.py`; their checksums are verified.
The archive retains all 60 fits' predictions, metrics and configuration evidence.

## Validation and practical limits

On Linux, acquire the pinned AttSiOff source for its upstream stopping regression
(`bash benchmark/competitors/tools/attsioff/setup.sh`, without `--docker`), then:

```bash
uv run --locked pytest -q
```

The CPU suite covers grouping, input identities, continuous metrics, stopping,
run isolation, features and paired analyses. Real graph/Torch checks live in
`benchmark/revision/tests/runtime_*.py` and need their respective model assets
and containers. Dataset reconstruction also runs on macOS. Exact numerical replay and Docker runtime
checks assume GNU/Linux utilities. Fresh uv verification and exact dataset
rebuild results for this publication are recorded in `provenance/github-v2-verification.json`.

Refitting is distinct from reproducing sealed predictions. Hardware and
floating-point differences can change results. The archived numerical evidence
is preserved, and no full benchmark retraining is claimed by a packaging check.
