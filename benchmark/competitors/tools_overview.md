# Tool Overview, Hyperparameters, and Validation Usage

This note summarizes each competitor tool at a high level, the hyperparameters
used in this repository, and how validation is used. Hyperparameters are aligned
to the paper descriptions and documented deviations (see `tools/<tool>.pdf` and
`TOOLS_CHANGES.md`).

All test evaluations report regression metrics from `scripts/metrics.py`:
MAE, MSE, RMSE, R2, Pearson, Spearman.

## Oligoformer

**Brief:** Sequence + RNA-FM embedding model for siRNA–target efficacy.
**Paper best features:** siRNA seq + mRNA seq + RNA‑FM embeddings + thermodynamic (TD) parameters.
**Inputs:** 19‑nt siRNA + 57‑nt extended_mRNA; RNA‑FM embeddings.
**Expected vs actual lengths:** expected 19/57; actual 19/57.
**Orientation:** siRNA antisense (guide) as provided; target window corresponds to the binding site.
**Alphabet:** RNA (U; T→U normalization in prepare).
**Required inputs/artifacts:** CSV (`siRNA`, `extended_mRNA`, `efficacy`); RNA‑FM embeddings generated during prepare.
**Prepared outputs:** `data/oligoformer/{train,val,test,leftout}.csv` (+ RNA‑FM embedding cache under `data/oligoformer/`).
**Hyperparameters (repo):** batch 16, lr 1e‑4, epochs 200, exp lr gamma 0.999,
early stopping 30, seed 42.
**Validation usage:** validation set is required and used for early stopping /
model selection. Optimized metric: validation loss (MSE).

## AttSiOff

**Brief:** Transformer encoder with RNA‑FM embeddings and handcrafted features.
**Inputs:** 19‑nt siRNA + 57‑nt extended_mRNA; RNA‑FM embeddings; optional prior
features (`s-Biopredsi`, `DSIR`, `i-score`, filled with 0.0 if missing).
**Paper best features:** RNA‑FM (antisense + local mRNA) + prior features (thermo, k‑mers, PSSM, structure, GC).
**Expected vs actual lengths:** paper 21/59; actual 19/57 (embeddings are padded/truncated to match model widths).
**Orientation:** antisense siRNA + local mRNA target window around binding site.
**Alphabet:** RNA (U; T→U normalization in prepare).
**Required inputs/artifacts:** CSV (`siRNA`, `extended_mRNA`, `efficacy`); optional `s-Biopredsi/DSIR/i-score`; RNA‑FM embeddings generated during prepare.
**Prepared outputs:** `data/attsioff/{train,val,test,leftout}.csv` (+ RNA‑FM embedding cache under `data/attsioff/`).
**Mismatches/notes:** paper uses 21‑nt antisense and 59‑nt local mRNA; benchmark uses 19/57.
**Hyperparameters (paper/repo):** batch 128, lr 0.005, epochs 1000, early stop
20 on PCC, loss MSE; encoder d_model 16, 4 layers, 2 heads; FC 256‑64‑16‑1.
**Validation usage:** validation set is required; used for early stopping.
Optimized metric: validation loss (MSE).

## GNN4siRNA

**Brief:** Graph neural network over k‑mer and thermodynamic features.
**Inputs:** 19‑nt siRNA + 57‑nt extended_mRNA; k‑mers (siRNA 3‑mer, mRNA 4‑mer),
RNAup/ViennaRNA thermodynamic features.
**Paper best features:** siRNA 3‑mers + mRNA 4‑mers + thermodynamic features (RNAup/ViennaRNA).
**Expected vs actual lengths:** paper uses full‑length mRNA; actual uses 57‑nt target window.
**Orientation:** siRNA antisense as provided; RNAup features use reverse‑complement conventions internally.
**Alphabet:** DNA for k‑mers/feature tooling (U→T).
**Required inputs/artifacts:** CSV (`siRNA`, `extended_mRNA`, `efficacy`); processed directory with k‑mers + thermo features.
**Prepared outputs:** `data/gnn4sirna/{train,val,test,leftout}.csv` and `data/gnn4sirna/processed/*`.
**Mismatches/notes:** paper expects full‑length mRNA; benchmark uses 57‑nt window.
**Hyperparameters (paper/repo):** batch 60, hop [8,4], layers [32,16],
dropout 0.15, Adamax lr 1e‑3, loss MSE, epochs 10.
**Validation usage:** validation set is required for training and model
selection. Optimized metric: validation loss (MSE).

## siRNADiscovery

**Brief:** Hybrid model combining thermodynamic, k‑mer, and AGO2-derived features.
**Inputs:** 19‑nt siRNA + 57‑nt extended_mRNA; preprocessing matrices and AGO2
features; thermodynamic and k‑mer features.
**Paper best features:** one‑hot + positional encoding + thermo/cofold/self‑fold + AGO2 + GC + 1–5‑mers + rule scores (with reduced matrices).
**Expected vs actual lengths:** paper uses full‑length mRNA (and longer siRNA); actual uses 19/57.
**Orientation:** siRNA antisense; target window reverse‑complement alignment is used to locate binding positions.
**Alphabet:** DNA for some upstream feature pipelines (U→T).
**Required inputs/artifacts:** CSV (`siRNA`, `extended_mRNA`, `efficacy`) plus `siRNA_split_preprocess` and `RNA_AGO2` directories.
**Prepared outputs:** `data/sirnadiscovery/{train,val,test,leftout}.csv` plus prepared feature matrices.
**Mismatches/notes:** upstream preprocess/AGO2 matrices may not cover benchmark IDs (missing rows filled with 0.0 in wrappers).
**Hyperparameters (paper/repo):** batch 64, lr 1e‑3, epochs 26, loss MSE,
HinSAGE [64,32], hop [12,6], dropout 0.1, dmodel 6; reduced dims:
mRNA=100, siRNA=6, pair=50.
**Validation usage:** validation set is required and used for model selection.
Optimized metric: validation loss (MSE).

## siRNABERT (BERT‑siRNA)

**Brief:** DNABERT 6‑mer token model over siRNA sequence only.
**Inputs:** 19‑nt siRNA (6‑mer tokens; target not used).
**Paper best features:** DNABERT 6‑mer tokenization; CLS embedding fed to an MLP.
**Expected vs actual lengths:** expected 19‑nt siRNA; actual 19‑nt siRNA.
**Orientation:** siRNA sequence as provided.
**Alphabet:** DNA tokens (U→T before tokenizing).
**Required inputs/artifacts:** CSV (`siRNA`, `efficacy`) only.
**Prepared outputs:** `data/sirnabert/{train,val,test,leftout}.csv`.
**Hyperparameters (repo):** epochs 30, batch 100, lr 5e‑5, max_len 16, seed 42.
**Validation usage:** validation set is required and used for model selection.
Optimized metric: validation loss (MSE).

## ENsiRNA

**Brief:** Multimodal geometric GNN using 3D structure + thermodynamic features.
**Inputs:** siRNA (anti/sense) + 57‑nt target window; PDB structures via Rosetta;
RNA‑FM embeddings; thermodynamic features.
**Paper best features:** multimodal geometric GNN combining RNA language model embeddings + structure + thermodynamics/modification features; 3D structures via Rosetta (FARFAR2‑style workflows).
**Expected vs actual lengths:** paper uses longer targets (e.g., 61‑nt window / full‑length); actual uses 57‑nt window.
**Orientation:** uses anti/sense siRNA strands; target position is tracked relative to the antisense index.
**Alphabet:** RNA (U; T→U normalization in prepare).
**Required inputs/artifacts:** JSONL with fields including `pdb_data_path`, `mRNA_seq`, `sense seq`, `anti seq`, `position`, `efficacy`; requires Rosetta to generate PDBs when missing.
**Prepared outputs:** `data/ensirna/{train,val,test,leftout}.jsonl` plus `data/ensirna/pdb/*.pdb`.
**Mismatches/notes:** paper uses longer target context; benchmark uses 57‑nt window.
**Hyperparameters (repo / paper config):** batch 16, lr 1e‑4, final_lr 1e‑5,
max_epoch 100. Wrapper defaults add `embed_dim=128` and `num_workers=0` for
stability in Docker.
**Validation usage:** validation set is required; checkpoint selection uses the
lowest validation loss (top‑k map) unless `--ensemble` is used at test time.
Optimized metric: validation loss (MSE).
