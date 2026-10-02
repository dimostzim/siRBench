# siRBench models

- `agentomics/fold_0` through `fold_4`: five selected Agentomics pipelines.
- `tabpfn/`: frozen and fine-tuned TabPFN-3.5.

Saved models are available in [Zenodo version 3](https://doi.org/10.5281/zenodo.23092108).
TabPFN fits are split into grouped and random archives, with a separate shared
foundation checkpoint archive.

Each model directory has `train.py`, `inference.py`, `pyproject.toml` and `uv.lock`.
Use Linux x86-64. TabPFN and Agentomics fold_0/fold_2 training require an NVIDIA
GPU. All Agentomics models support CPU inference.

## Agentomics setup

From the repository root:

```bash
git lfs install
git lfs pull
uv sync --frozen --project models/agentomics/fold_0
```

Replace `fold_0` with the desired fold. Each directory includes its fitted
`training_artifacts/` and the feature/model modules used by training and inference.
No Agentomics or LLM account is needed.

## Agentomics training

Prepare the data using [data/](../data/README.md), then:

```bash
uv run --frozen --project models/agentomics/fold_0 \
  python models/agentomics/fold_0/train.py \
  --train-data data/processed/reference-training/fold_0/train \
  --validation-data data/processed/reference-training/fold_0/validation \
  --artifacts-dir outputs/fold_0/training_artifacts
```

Training directories contain `input/data.csv` and `labels.csv` with `id,label`.
Training refits the selected pipeline for one grouped fold. The five models
are used independently. The Agentomics search is not repeated.

## Agentomics inference

```bash
mkdir -p outputs
uv run --frozen --project models/agentomics/fold_0 \
  python models/agentomics/fold_0/inference.py \
  --input data/processed/reference-training/fold_0/test/input \
  --output outputs/fold_0_predictions.csv
```

The input directory contains `data.csv` with `id`, 19-nt antisense `siRNA` and
57-nt `extended_mRNA`. No labels or source/cell-line covariates are used.
Output columns are `id,prediction`. Add `--artifacts-dir` to use a refitted model.

## TabPFN setup and training

```bash
uv sync --locked --project models/tabpfn
export R="$PWD/data/archive/siRBench-v2-2026-09-27/workspace"
```

Obtain the official checkpoint through authorized
[Prior Labs access](https://huggingface.co/Prior-Labs/tabpfn_3_5) and place it at
`$R/evaluation/tabpfn-v1/models/v3.5/tabpfn-v3.5-20260909.safetensors`.
The required SHA256 is
`ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`.
The locked environment and runner preserve the published settings.

```bash
uv run --locked --project models/tabpfn python models/tabpfn/train.py \
  --root "$R" --axis grouped --fold 0 --seed 0 \
  --output "$PWD/outputs/tabpfn-grouped-0-seed-0"
```

This writes both `tabpfn35_frozen/` and `tabpfn35_finetuned/` models. Use a new
output directory. The full matrix uses `grouped`/`random`, folds 0–4 and seeds 0–2.

## TabPFN inference

```bash
uv run --locked --project models/tabpfn python models/tabpfn/inference.py \
  --root "$R" \
  --model-dir outputs/tabpfn-grouped-0-seed-0/tabpfn35_frozen \
  --partition grouped/fold_0/test.csv \
  --output outputs/tabpfn-predictions
```

Use `tabpfn35_finetuned` for the fine-tuned model. For new featurized records,
replace `--partition` with `--input features.csv`, containing `record_id`, `siRNA`
and the 100 columns in the archived feature manifest. The model uses these plus
76 guide one-hot features.

For downloaded fits, pass their directory with `--model-dir`. Fine-tuned fits
expect `selected_weights.pth` alongside `model.tabpfn_fit` and `train_meta.json`.
Use `--weights /path/to/checkpoint` to specify a different location, including
the foundation checkpoint for frozen fits. Checksums are verified before loading.
TabPFN checkpoints follow the [Prior Labs license](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE).
Cross-GPU reloads may have small floating-point differences.
