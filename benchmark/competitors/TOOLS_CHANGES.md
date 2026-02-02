# Tool Changes vs Upstream

Global decisions

- Use 19-nt siRNA and 57-nt target (extended_mRNA) for all tools.
- Sequences are provided 5'->3' for both siRNA and target; siRNA is antisense and matches the target by Watson-Crick pairing.
- When papers use longer targets (AttSiOff 59 nt, ENsiRNA 61 nt, GNN4siRNA/siRNADiscovery full-length), we document the deviation below.
- Reproducibility: all wrappers accept `--seed` (default 42) and `--deterministic` for best-effort reproducibility; ENsiRNA upstream uses a fixed seed (12) and deterministic algorithms.
  - Example: `python3 benchmark/competitors/scripts/train.py --tool oligoformer --seed 42 --deterministic ...`
- `benchmark/competitors/run_tool.sh` skips training if the model file exists, and skips testing if the metrics JSON exists (delete outputs to force reruns).

## Oligoformer

- Inputs: 19-nt siRNA + 57-nt target (paper uses 57; no deviation).
- Hyperparams: lr 1e-4, batch 16, epochs 200, exp lr gamma 0.999, early stop 30 (repo defaults; paper does not specify in wrapper).
- Pipeline: `benchmark/competitors/run_tool.sh` passes these explicitly.
- Reproducibility: wrapper seeds Python/NumPy/Torch; `--deterministic` enables cudnn + deterministic algorithms.

## siRNABERT

- Inputs: 19-nt siRNA only with 6-mer tokenization; target not used (matches paper).
- Hyperparams: epochs 30, batch 100, lr 5e-5, max_len 16 (repo defaults).
- Reproducibility: wrapper seeds Python/NumPy/Torch; `--deterministic` enables cudnn + deterministic algorithms.

## AttSiOff

- Inputs: 19-nt siRNA, 57-nt target (paper uses 21/59; deviation).
- RNA-FM embeddings are padded/truncated to 21/59 to match the original model input width (21*4 + 59*1 + 110).
- Optional columns `s-Biopredsi`, `DSIR`, `i-score` are filled with 0.0 if missing (the model does not consume them).
- Hyperparams: batch 128, lr 0.005, epochs 1000, early stopping 20 (paper defaults).
- Validation: optimize Spearman (SPCC) for early stopping (matches upstream).
- Reproducibility: wrapper seeds Python/NumPy/Torch; `--deterministic` enables cudnn + deterministic algorithms.

## GNN4siRNA

- Inputs: 19-nt siRNA, 57-nt target (paper uses full-length mRNA; deviation).
- Preprocess uses k_sirna=3, k_mrna=4; hop [8,4], layer [32,16], dropout 0.15 (paper).
- Optimizer: Adamax lr 1e-3 (paper); epochs 10 (repo default).
- Stable IDs use sequence hashes to avoid collisions across splits.
- Preprocess is done per split (train/val and test/leftout separately) for strict inductive evaluation.
- Reproducibility: wrapper seeds Python/NumPy/TF; `--deterministic` enables TF op determinism when available.

## siRNADiscovery

- Inputs: 19-nt siRNA, 57-nt target (paper uses full-length mRNA; deviation).
- Params updated: sirna_length 19, max_mrna_len 57; other params match paper (dmodel 6, batch 64, epochs 26, hop [12,6], layers [64,32], dropout 0.1, lr 1e-3, MSE).
- Stable IDs use sequence hashes to avoid collisions.
- Preprocess matrices (RNAfold/RNAcofold) can be generated via `prepare.py --run-preprocess` (ViennaRNA required). AGO2/RPISeq features remain external inputs.
- AGO2 features can be provided from `data/sirnadiscovery/RNA_AGO2`; the wrapper avoids copying if the source and destination are the same.
- Preprocess is done per split (train/val and test/leftout separately) for strict inductive evaluation.
- Reproducibility: wrapper seeds Python/NumPy/TF; `--deterministic` enables TF op determinism when available.

## ENsiRNA

- Inputs: 19-nt siRNA, 57-nt target (paper uses 61-nt target; deviation).
- Rosetta is extracted from `rosettacommons/rosetta` and baked into the ENsiRNA image at `/opt/rosetta` (via `fetch_rosetta.sh`).
- Hyperparams: batch 16, lr 1e-4, final_lr 1e-5, max_epoch 100 (config.json).
- Dataset patch: `sec_pos`/`chain` are regenerated in `dataset.py` to match the assembled `S` structure (fixes JSONL mismatch); requires regenerating processed caches if inputs change.
- Training wrapper defaults: `--embed_dim 128` (matches feature width) and `--num_workers 0` (avoids Docker shm crashes).
- Testing: default to the best checkpoint (lowest val loss from `topk_map.txt`); use `--ensemble` to average multiple checkpoints.
- Test patch: tolerate missing `siRNA` in JSONL by falling back to `id` or a row id.
- Seed patch: `ENSIRNA_SEED` env var controls the upstream seed (default 12).
- Reproducibility: upstream ENsiRNA sets a fixed seed (12) and enables deterministic algorithms; wrapper does not override.
