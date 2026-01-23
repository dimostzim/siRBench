#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Parse flags
RUN_OLIGOFORMER=0
RUN_SIRNADISCOVERY=0
RUN_SIRNABERT=0
RUN_ATTSIOFF=0
RUN_GNN4SIRNA=0
RUN_ENSIRNA=0
RUN_ALL=1

for arg in "$@"; do
    case $arg in
        --oligoformer) RUN_OLIGOFORMER=1; RUN_ALL=0 ;;
        --sirnadiscovery) RUN_SIRNADISCOVERY=1; RUN_ALL=0 ;;
        --sirnabert) RUN_SIRNABERT=1; RUN_ALL=0 ;;
        --attsioff) RUN_ATTSIOFF=1; RUN_ALL=0 ;;
        --gnn4sirna) RUN_GNN4SIRNA=1; RUN_ALL=0 ;;
        --ensirna) RUN_ENSIRNA=1; RUN_ALL=0 ;;
        --help|-h)
            echo "Usage: $0 [--oligoformer] [--sirnadiscovery] [--sirnabert] [--attsioff] [--gnn4sirna] [--ensirna]"
            echo "If no flags provided, runs all tools."
            exit 0 ;;
    esac
done

if [ "$RUN_ALL" = "1" ]; then
    RUN_OLIGOFORMER=1
    RUN_SIRNADISCOVERY=1
    RUN_SIRNABERT=1
    RUN_ATTSIOFF=1
    RUN_GNN4SIRNA=1
    RUN_ENSIRNA=1
fi

DATA_ROOT="${DATA_ROOT:-../data}"
TRAIN_CSV="${DATA_ROOT}/siRBench_full_base_1000_train.csv"
VAL_CSV="${DATA_ROOT}/siRBench_full_base_1000_val.csv"
TEST_CSV="${DATA_ROOT}/siRBench_full_base_1000_test.csv"

mkdir -p results

# ============ OLIGOFORMER ============
if [ "$RUN_OLIGOFORMER" = "1" ]; then
    echo "=== OLIGOFORMER ==="
    mkdir -p data/oligoformer models/oligoformer results/oligoformer

    RNAFM_ROOT="tools/oligoformer/oligoformer_src/RNA-FM"

    python3 prepare.py --tool oligoformer --input-csv "$TRAIN_CSV" --output-dir data/oligoformer --dataset-name train --run-rnafm --rnafm-root "$RNAFM_ROOT"
    python3 prepare.py --tool oligoformer --input-csv "$VAL_CSV" --output-dir data/oligoformer --dataset-name val --run-rnafm --rnafm-root "$RNAFM_ROOT"
    python3 prepare.py --tool oligoformer --input-csv "$TEST_CSV" --output-dir data/oligoformer --dataset-name test --run-rnafm --rnafm-root "$RNAFM_ROOT"

    python3 train.py --tool oligoformer --train-csv data/oligoformer/train.csv --val-csv data/oligoformer/val.csv --data-dir data/oligoformer --model-dir models/oligoformer \
        --epochs 200 --batch-size 16 --lr 1e-4 --weight-decay 0.999 --early-stopping 30 --seed 42

    python3 test.py --tool oligoformer --test-csv data/oligoformer/test.csv --data-dir data/oligoformer --model-path models/oligoformer/model.pt --output-csv results/oligoformer/preds.csv --metrics-json results/oligoformer/metrics.json
fi

# ============ SIRNADISCOVERY ============
if [ "$RUN_SIRNADISCOVERY" = "1" ]; then
    echo "=== SIRNADISCOVERY ==="
    mkdir -p data/sirnadiscovery models/sirnadiscovery results/sirnadiscovery

    PREPROCESS_DIR="tools/sirnadiscovery/sirnadiscovery_src/siRNA_split/siRNA_split_preprocess"
    RNA_AGO2_DIR="tools/sirnadiscovery/sirnadiscovery_src/siRNA_split/RNA_AGO2"

    python3 prepare.py --tool sirnadiscovery --input-csv "$TRAIN_CSV" --output-dir data/sirnadiscovery --dataset-name train --preprocess-dir "$PREPROCESS_DIR" --rna-ago2-dir "$RNA_AGO2_DIR" --mrna-col extended_mRNA
    python3 prepare.py --tool sirnadiscovery --input-csv "$VAL_CSV" --output-dir data/sirnadiscovery --dataset-name val --preprocess-dir "$PREPROCESS_DIR" --rna-ago2-dir "$RNA_AGO2_DIR" --mrna-col extended_mRNA
    python3 prepare.py --tool sirnadiscovery --input-csv "$TEST_CSV" --output-dir data/sirnadiscovery --dataset-name test --preprocess-dir "$PREPROCESS_DIR" --rna-ago2-dir "$RNA_AGO2_DIR" --mrna-col extended_mRNA

    python3 train.py --tool sirnadiscovery --train-csv data/sirnadiscovery/train.csv --val-csv data/sirnadiscovery/val.csv --preprocess-dir "$PREPROCESS_DIR" --rna-ago2-dir "$RNA_AGO2_DIR" --model-dir models/sirnadiscovery

    python3 test.py --tool sirnadiscovery --test-csv data/sirnadiscovery/test.csv --preprocess-dir "$PREPROCESS_DIR" --rna-ago2-dir "$RNA_AGO2_DIR" --model-path models/sirnadiscovery/model.keras --output-csv results/sirnadiscovery/preds.csv --metrics-json results/sirnadiscovery/metrics.json
fi

# ============ SIRNABERT ============
if [ "$RUN_SIRNABERT" = "1" ]; then
    echo "=== SIRNABERT ==="
    mkdir -p data/sirnabert models/sirnabert results/sirnabert

    python3 prepare.py --tool sirnabert --input-csv "$TRAIN_CSV" --output-dir data/sirnabert --dataset-name train
    python3 prepare.py --tool sirnabert --input-csv "$VAL_CSV" --output-dir data/sirnabert --dataset-name val
    python3 prepare.py --tool sirnabert --input-csv "$TEST_CSV" --output-dir data/sirnabert --dataset-name test

    python3 train.py --tool sirnabert --train-csv data/sirnabert/train.csv --val-csv data/sirnabert/val.csv --model-dir models/sirnabert \
        --epochs 30 --batch-size 100 --lr 5e-5 --max-len 16 --seed 42

    python3 test.py --tool sirnabert --test-csv data/sirnabert/test.csv --model-path models/sirnabert/model.pt --output-csv results/sirnabert/preds.csv --metrics-json results/sirnabert/metrics.json
fi

# ============ ATTSIOFF ============
if [ "$RUN_ATTSIOFF" = "1" ]; then
    echo "=== ATTSIOFF ==="
    mkdir -p data/attsioff models/attsioff results/attsioff

    python3 prepare.py --tool attsioff --input-csv "$TRAIN_CSV" --output-dir data/attsioff --dataset-name train
    python3 prepare.py --tool attsioff --input-csv "$VAL_CSV" --output-dir data/attsioff --dataset-name val
    python3 prepare.py --tool attsioff --input-csv "$TEST_CSV" --output-dir data/attsioff --dataset-name test

    python3 train.py --tool attsioff --train-csv data/attsioff/train.csv --val-csv data/attsioff/val.csv --data-dir data/attsioff --model-dir models/attsioff \
        --batch-size 128 --epochs 1000 --lr 0.005 --early-stopping 20

    python3 test.py --tool attsioff --test-csv data/attsioff/test.csv --data-dir data/attsioff --model-path models/attsioff/model.pt --output-csv results/attsioff/preds.csv --metrics-json results/attsioff/metrics.json
fi

# ============ GNN4SIRNA ============
if [ "$RUN_GNN4SIRNA" = "1" ]; then
    echo "=== GNN4SIRNA ==="
    mkdir -p data/gnn4sirna models/gnn4sirna results/gnn4sirna

    python3 prepare.py --tool gnn4sirna --input-csv "$TRAIN_CSV" --output-dir data/gnn4sirna --dataset-name train --mrna-col extended_mRNA
    python3 prepare.py --tool gnn4sirna --input-csv "$VAL_CSV" --output-dir data/gnn4sirna --dataset-name val --mrna-col extended_mRNA
    python3 prepare.py --tool gnn4sirna --input-csv "$TEST_CSV" --output-dir data/gnn4sirna --dataset-name test --mrna-col extended_mRNA
    python3 - <<'PY'
import csv

paths = [
    "../data/siRBench_full_base_1000_train.csv",
    "../data/siRBench_full_base_1000_val.csv",
    "../data/siRBench_full_base_1000_test.csv",
]
out_path = "data/gnn4sirna/all_input.csv"

with open(out_path, "w", newline="") as out_f:
    writer = None
    for path in paths:
        with open(path, newline="") as in_f:
            reader = csv.reader(in_f)
            header = next(reader)
            if writer is None:
                writer = csv.writer(out_f)
                writer.writerow(header)
            for row in reader:
                writer.writerow(row)
PY
    python3 prepare.py --tool gnn4sirna --input-csv data/gnn4sirna/all_input.csv --output-dir data/gnn4sirna --dataset-name all --mrna-col extended_mRNA

    python3 train.py --tool gnn4sirna --train-csv data/gnn4sirna/train.csv --val-csv data/gnn4sirna/val.csv --processed-dir data/gnn4sirna/processed/all --model-dir models/gnn4sirna \
        --batch-size 60 --epochs 10 --lr 1e-3 --loss mse

    python3 test.py --tool gnn4sirna --test-csv data/gnn4sirna/test.csv --processed-dir data/gnn4sirna/processed/all --model-path models/gnn4sirna/model.keras --output-csv results/gnn4sirna/preds.csv --metrics-json results/gnn4sirna/metrics.json
fi

# ============ ENSIRNA ============
if [ "$RUN_ENSIRNA" = "1" ]; then
    echo "=== ENSIRNA ==="
    mkdir -p data/ensirna models/ensirna results/ensirna

    python3 prepare.py --tool ensirna --input-csv "$TRAIN_CSV" --output-jsonl data/ensirna/train.jsonl
    python3 prepare.py --tool ensirna --input-csv "$VAL_CSV" --output-jsonl data/ensirna/val.jsonl
    python3 prepare.py --tool ensirna --input-csv "$TEST_CSV" --output-jsonl data/ensirna/test.jsonl

    python3 train.py --tool ensirna --train-set data/ensirna/train.jsonl --valid-set data/ensirna/val.jsonl --model-dir models/ensirna \
        --batch-size 16 --lr 1e-4 --final_lr 1e-5 --max_epoch 100

    CKPTS=(models/ensirna/*.ckpt)
    if [ ${#CKPTS[@]} -eq 0 ]; then
        echo "No ENsiRNA checkpoints found in models/ensirna; skipping test."
    else
        python3 test.py --tool ensirna --test-set data/ensirna/test.jsonl --ckpt "${CKPTS[@]}" \
            --save-dir results/ensirna --run-id ensirna \
            --output-csv results/ensirna/preds.csv --metrics-json results/ensirna/metrics.json
    fi
fi

echo "=== DONE ==="
echo "Results in results/*/"
