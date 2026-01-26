# Tool Overview, Hyperparameters, and Validation Usage

This note summarizes each competitor tool at a high level, the hyperparameters
used in this repository, and how validation is used. Hyperparameters are aligned
to the paper descriptions and documented deviations (see `tools/<tool>.pdf` and
`TOOLS_CHANGES.md`).

All test evaluations report regression metrics from `scripts/metrics.py`:
MAE, MSE, RMSE, R2, Pearson, Spearman.

## Oligoformer

**Brief:** Sequence + RNA-FM embedding model for siRNA–target efficacy.
**Inputs:** 19‑nt siRNA + 57‑nt extended_mRNA; RNA‑FM embeddings.
**Hyperparameters (repo):** batch 16, lr 1e‑4, epochs 200, exp lr gamma 0.999,
early stopping 30, seed 42.
**Validation usage:** validation set is required and used for early stopping /
model selection. Optimized metric: validation loss (MSE).

## AttSiOff

**Brief:** Transformer encoder with RNA‑FM embeddings and handcrafted features.
**Inputs:** 19‑nt siRNA + 57‑nt extended_mRNA; RNA‑FM embeddings; optional prior
features (`s-Biopredsi`, `DSIR`, `i-score`, filled with 0.0 if missing).
**Hyperparameters (paper/repo):** batch 128, lr 0.005, epochs 1000, early stop
20 on PCC, loss MSE; encoder d_model 16, 4 layers, 2 heads; FC 256‑64‑16‑1.
**Validation usage:** validation set is required; used for early stopping.
Optimized metric: validation loss (MSE).

## GNN4siRNA

**Brief:** Graph neural network over k‑mer and thermodynamic features.
**Inputs:** 19‑nt siRNA + 57‑nt extended_mRNA; k‑mers (siRNA 3‑mer, mRNA 4‑mer),
RNAup/ViennaRNA thermodynamic features.
**Hyperparameters (paper/repo):** batch 60, hop [8,4], layers [32,16],
dropout 0.15, Adamax lr 1e‑3, loss MSE, epochs 10.
**Validation usage:** validation set is required for training and model
selection. Optimized metric: validation loss (MSE).

## siRNADiscovery

**Brief:** Hybrid model combining thermodynamic, k‑mer, and AGO2-derived features.
**Inputs:** 19‑nt siRNA + 57‑nt extended_mRNA; preprocessing matrices and AGO2
features; thermodynamic and k‑mer features.
**Hyperparameters (paper/repo):** batch 64, lr 1e‑3, epochs 26, loss MSE,
HinSAGE [64,32], hop [12,6], dropout 0.1, dmodel 6; reduced dims:
mRNA=100, siRNA=6, pair=50.
**Validation usage:** validation set is required and used for model selection.
Optimized metric: validation loss (MSE).

## siRNABERT (BERT‑siRNA)

**Brief:** DNABERT 6‑mer token model over siRNA sequence only.
**Inputs:** 19‑nt siRNA (6‑mer tokens; target not used).
**Hyperparameters (repo):** epochs 30, batch 100, lr 5e‑5, max_len 16, seed 42.
**Validation usage:** validation set is required and used for model selection.
Optimized metric: validation loss (MSE).

## ENsiRNA

**Brief:** Multimodal geometric GNN using 3D structure + thermodynamic features.
**Inputs:** siRNA (anti/sense) + 57‑nt target window; PDB structures via Rosetta;
RNA‑FM embeddings; thermodynamic features.
**Hyperparameters (repo / paper config):** batch 16, lr 1e‑4, final_lr 1e‑5,
max_epoch 100. Wrapper defaults add `embed_dim=128` and `num_workers=0` for
stability in Docker.
**Validation usage:** validation set is required; checkpoint selection uses the
lowest validation loss (top‑k map) unless `--ensemble` is used at test time.
Optimized metric: validation loss (MSE).
