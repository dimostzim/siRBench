#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

ensure_dirs() {
    local tool="$1"
    shift
    if mkdir -p "$@"; then
        return
    fi
    echo "Failed to create output dirs for ${tool}: $*"
    echo "Remove root-owned dirs with Docker, then retry."
    exit 1
}

# Parse flags
RUN_OLIGOFORMER=0
RUN_SIRNADISCOVERY=0
RUN_SIRNABERT=0
RUN_ATTSIOFF=0
RUN_GNN4SIRNA=0
RUN_ENSIRNA=0
RUN_ALL=1
TRAIN_CSV=""
VAL_CSV=""
TEST_CSV=""
LEFTOUT_CSV=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --tool)
            shift
            if [ $# -eq 0 ] || [[ "$1" == --* ]]; then
                echo "Missing value for --tool"
                exit 1
            fi
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                TOOL="$1"
                case $TOOL in
                    oligoformer) RUN_OLIGOFORMER=1 ;;
                    sirnadiscovery) RUN_SIRNADISCOVERY=1 ;;
                    sirnabert) RUN_SIRNABERT=1 ;;
                    attsioff) RUN_ATTSIOFF=1 ;;
                    gnn4sirna) RUN_GNN4SIRNA=1 ;;
                    ensirna) RUN_ENSIRNA=1 ;;
                    *)
                        echo "Unknown tool: $TOOL"
                        exit 1 ;;
                esac
                RUN_ALL=0
                shift
            done
            ;;
        --train)
            shift
            if [ $# -eq 0 ] || [[ "$1" == --* ]]; then
                echo "Missing value for --train"
                exit 1
            fi
            TRAIN_CSV="$1"
            shift
            ;;
        --val)
            shift
            if [ $# -eq 0 ] || [[ "$1" == --* ]]; then
                echo "Missing value for --val"
                exit 1
            fi
            VAL_CSV="$1"
            shift
            ;;
        --test)
            shift
            if [ $# -eq 0 ] || [[ "$1" == --* ]]; then
                echo "Missing value for --test"
                exit 1
            fi
            TEST_CSV="$1"
            shift
            ;;
        --leftout)
            shift
            if [ $# -eq 0 ] || [[ "$1" == --* ]]; then
                echo "Missing value for --leftout"
                exit 1
            fi
            LEFTOUT_CSV="$1"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--tool <name>...] --train <path> --val <path> --test <path> [--leftout <path>]"
            echo "If no flags provided, runs all tools. --leftout runs an extra unseen test set."
            exit 0 ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
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

if [ -z "${TRAIN_CSV}" ] || [ -z "${VAL_CSV}" ] || [ -z "${TEST_CSV}" ]; then
    echo "Missing required inputs. Provide --train, --val, and --test."
    exit 1
fi

# ============ OLIGOFORMER ============
if [ "$RUN_OLIGOFORMER" = "1" ]; then
    echo "=== OLIGOFORMER ==="
    ensure_dirs oligoformer data/oligoformer models/oligoformer results/oligoformer

    RNAFM_ROOT="tools/oligoformer/oligoformer_src/RNA-FM"

    python3 prepare.py --tool oligoformer --input-csv "$TRAIN_CSV" --output-dir data/oligoformer --dataset-name train --run-rnafm --rnafm-root "$RNAFM_ROOT"
    python3 prepare.py --tool oligoformer --input-csv "$VAL_CSV" --output-dir data/oligoformer --dataset-name val --run-rnafm --rnafm-root "$RNAFM_ROOT"
    python3 prepare.py --tool oligoformer --input-csv "$TEST_CSV" --output-dir data/oligoformer --dataset-name test --run-rnafm --rnafm-root "$RNAFM_ROOT"
    if [ -n "${LEFTOUT_CSV}" ]; then
        python3 prepare.py --tool oligoformer --input-csv "$LEFTOUT_CSV" --output-dir data/oligoformer --dataset-name leftout --run-rnafm --rnafm-root "$RNAFM_ROOT"
    fi

    python3 scripts/train.py --tool oligoformer --train-csv data/oligoformer/train.csv --val-csv data/oligoformer/val.csv --data-dir data/oligoformer --model-dir models/oligoformer \
        --epochs 200 --batch-size 16 --lr 1e-4 --weight-decay 0.999 --early-stopping 30 --seed 42

    python3 scripts/test.py --tool oligoformer --test-csv data/oligoformer/test.csv --data-dir data/oligoformer --model-path models/oligoformer/model.pt --output-csv results/oligoformer/preds.csv --metrics-json results/oligoformer/metrics.json
    if [ -n "${LEFTOUT_CSV}" ]; then
        python3 scripts/test.py --tool oligoformer --test-csv data/oligoformer/leftout.csv --data-dir data/oligoformer --model-path models/oligoformer/model.pt --output-csv results/oligoformer/preds_leftout.csv --metrics-json results/oligoformer/metrics_leftout.json
    fi
fi

# ============ SIRNADISCOVERY ============
if [ "$RUN_SIRNADISCOVERY" = "1" ]; then
    echo "=== SIRNADISCOVERY ==="
    ensure_dirs sirnadiscovery data/sirnadiscovery models/sirnadiscovery results/sirnadiscovery

    PREPROCESS_DIR="tools/sirnadiscovery/sirnadiscovery_src/siRNA_split/siRNA_split_preprocess"
    RNA_AGO2_DIR="tools/sirnadiscovery/sirnadiscovery_src/siRNA_split/RNA_AGO2"

    python3 prepare.py --tool sirnadiscovery --input-csv "$TRAIN_CSV" --output-dir data/sirnadiscovery --dataset-name train --preprocess-dir "$PREPROCESS_DIR" --rna-ago2-dir "$RNA_AGO2_DIR" --mrna-col extended_mRNA
    python3 prepare.py --tool sirnadiscovery --input-csv "$VAL_CSV" --output-dir data/sirnadiscovery --dataset-name val --preprocess-dir "$PREPROCESS_DIR" --rna-ago2-dir "$RNA_AGO2_DIR" --mrna-col extended_mRNA
    python3 prepare.py --tool sirnadiscovery --input-csv "$TEST_CSV" --output-dir data/sirnadiscovery --dataset-name test --preprocess-dir "$PREPROCESS_DIR" --rna-ago2-dir "$RNA_AGO2_DIR" --mrna-col extended_mRNA
    if [ -n "${LEFTOUT_CSV}" ]; then
        python3 prepare.py --tool sirnadiscovery --input-csv "$LEFTOUT_CSV" --output-dir data/sirnadiscovery --dataset-name leftout --preprocess-dir "$PREPROCESS_DIR" --rna-ago2-dir "$RNA_AGO2_DIR" --mrna-col extended_mRNA
    fi

    python3 scripts/train.py --tool sirnadiscovery --train-csv data/sirnadiscovery/train.csv --val-csv data/sirnadiscovery/val.csv --preprocess-dir "$PREPROCESS_DIR" --rna-ago2-dir "$RNA_AGO2_DIR" --model-dir models/sirnadiscovery

    python3 scripts/test.py --tool sirnadiscovery --test-csv data/sirnadiscovery/test.csv --preprocess-dir "$PREPROCESS_DIR" --rna-ago2-dir "$RNA_AGO2_DIR" --model-path models/sirnadiscovery/model.keras --output-csv results/sirnadiscovery/preds.csv --metrics-json results/sirnadiscovery/metrics.json
    if [ -n "${LEFTOUT_CSV}" ]; then
        python3 scripts/test.py --tool sirnadiscovery --test-csv data/sirnadiscovery/leftout.csv --preprocess-dir "$PREPROCESS_DIR" --rna-ago2-dir "$RNA_AGO2_DIR" --model-path models/sirnadiscovery/model.keras --output-csv results/sirnadiscovery/preds_leftout.csv --metrics-json results/sirnadiscovery/metrics_leftout.json
    fi
fi

# ============ SIRNABERT ============
if [ "$RUN_SIRNABERT" = "1" ]; then
    echo "=== SIRNABERT ==="
    ensure_dirs sirnabert data/sirnabert models/sirnabert results/sirnabert

    python3 prepare.py --tool sirnabert --input-csv "$TRAIN_CSV" --output-dir data/sirnabert --dataset-name train
    python3 prepare.py --tool sirnabert --input-csv "$VAL_CSV" --output-dir data/sirnabert --dataset-name val
    python3 prepare.py --tool sirnabert --input-csv "$TEST_CSV" --output-dir data/sirnabert --dataset-name test
    if [ -n "${LEFTOUT_CSV}" ]; then
        python3 prepare.py --tool sirnabert --input-csv "$LEFTOUT_CSV" --output-dir data/sirnabert --dataset-name leftout
    fi

    python3 scripts/train.py --tool sirnabert --train-csv data/sirnabert/train.csv --val-csv data/sirnabert/val.csv --model-dir models/sirnabert \
        --epochs 30 --batch-size 100 --lr 5e-5 --max-len 16 --seed 42

    python3 scripts/test.py --tool sirnabert --test-csv data/sirnabert/test.csv --model-path models/sirnabert/model.pt --output-csv results/sirnabert/preds.csv --metrics-json results/sirnabert/metrics.json
    if [ -n "${LEFTOUT_CSV}" ]; then
        python3 scripts/test.py --tool sirnabert --test-csv data/sirnabert/leftout.csv --model-path models/sirnabert/model.pt --output-csv results/sirnabert/preds_leftout.csv --metrics-json results/sirnabert/metrics_leftout.json
    fi
fi

# ============ ATTSIOFF ============
if [ "$RUN_ATTSIOFF" = "1" ]; then
    echo "=== ATTSIOFF ==="
    ensure_dirs attsioff data/attsioff models/attsioff results/attsioff

    python3 prepare.py --tool attsioff --input-csv "$TRAIN_CSV" --output-dir data/attsioff --dataset-name train
    python3 prepare.py --tool attsioff --input-csv "$VAL_CSV" --output-dir data/attsioff --dataset-name val
    python3 prepare.py --tool attsioff --input-csv "$TEST_CSV" --output-dir data/attsioff --dataset-name test
    if [ -n "${LEFTOUT_CSV}" ]; then
        python3 prepare.py --tool attsioff --input-csv "$LEFTOUT_CSV" --output-dir data/attsioff --dataset-name leftout
    fi

    python3 scripts/train.py --tool attsioff --train-csv data/attsioff/train.csv --val-csv data/attsioff/val.csv --data-dir data/attsioff --model-dir models/attsioff \
        --batch-size 128 --epochs 1000 --lr 0.005 --early-stopping 20

    python3 scripts/test.py --tool attsioff --test-csv data/attsioff/test.csv --data-dir data/attsioff --model-path models/attsioff/model.pt --output-csv results/attsioff/preds.csv --metrics-json results/attsioff/metrics.json
    if [ -n "${LEFTOUT_CSV}" ]; then
        python3 scripts/test.py --tool attsioff --test-csv data/attsioff/leftout.csv --data-dir data/attsioff --model-path models/attsioff/model.pt --output-csv results/attsioff/preds_leftout.csv --metrics-json results/attsioff/metrics_leftout.json
    fi
fi

# ============ GNN4SIRNA ============
if [ "$RUN_GNN4SIRNA" = "1" ]; then
    echo "=== GNN4SIRNA ==="
    ensure_dirs gnn4sirna data/gnn4sirna models/gnn4sirna results/gnn4sirna

    python3 prepare.py --tool gnn4sirna --input-csv "$TRAIN_CSV" --output-dir data/gnn4sirna --dataset-name train --mrna-col extended_mRNA
    python3 prepare.py --tool gnn4sirna --input-csv "$VAL_CSV" --output-dir data/gnn4sirna --dataset-name val --mrna-col extended_mRNA
    python3 prepare.py --tool gnn4sirna --input-csv "$TEST_CSV" --output-dir data/gnn4sirna --dataset-name test --mrna-col extended_mRNA
    if [ -n "${LEFTOUT_CSV}" ]; then
        python3 prepare.py --tool gnn4sirna --input-csv "$LEFTOUT_CSV" --output-dir data/gnn4sirna --dataset-name leftout --mrna-col extended_mRNA
    fi
python3 - <<PY
import csv

paths = [
    "${TRAIN_CSV}",
    "${VAL_CSV}",
    "${TEST_CSV}",
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

    python3 scripts/train.py --tool gnn4sirna --train-csv data/gnn4sirna/train.csv --val-csv data/gnn4sirna/val.csv --processed-dir data/gnn4sirna/processed/all --model-dir models/gnn4sirna \
        --batch-size 60 --epochs 10 --lr 1e-3 --loss mse

    python3 scripts/test.py --tool gnn4sirna --test-csv data/gnn4sirna/test.csv --processed-dir data/gnn4sirna/processed/all --model-path models/gnn4sirna/model.keras --output-csv results/gnn4sirna/preds.csv --metrics-json results/gnn4sirna/metrics.json
    if [ -n "${LEFTOUT_CSV}" ]; then
        python3 scripts/test.py --tool gnn4sirna --test-csv data/gnn4sirna/leftout.csv --processed-dir data/gnn4sirna/processed/leftout --model-path models/gnn4sirna/model.keras --output-csv results/gnn4sirna/preds_leftout.csv --metrics-json results/gnn4sirna/metrics_leftout.json
    fi
fi

# ============ ENSIRNA ============
if [ "$RUN_ENSIRNA" = "1" ]; then
    echo "=== ENSIRNA ==="
    ensure_dirs ensirna data/ensirna models/ensirna results/ensirna

    python3 prepare.py --tool ensirna --input-csv "$TRAIN_CSV" --output-jsonl data/ensirna/train.jsonl
    python3 prepare.py --tool ensirna --input-csv "$VAL_CSV" --output-jsonl data/ensirna/val.jsonl
    python3 prepare.py --tool ensirna --input-csv "$TEST_CSV" --output-jsonl data/ensirna/test.jsonl
    if [ -n "${LEFTOUT_CSV}" ]; then
        python3 prepare.py --tool ensirna --input-csv "$LEFTOUT_CSV" --output-jsonl data/ensirna/leftout.jsonl
    fi

    python3 scripts/train.py --tool ensirna --train-set data/ensirna/train.jsonl --valid-set data/ensirna/val.jsonl --model-dir models/ensirna \
        --batch-size 16 --lr 1e-4 --final_lr 1e-5 --max_epoch 100

    shopt -s globstar nullglob
    CKPTS=(models/ensirna/**/*.ckpt)
    shopt -u globstar nullglob
    if [ ${#CKPTS[@]} -eq 0 ]; then
        echo "No ENsiRNA checkpoints found in models/ensirna; skipping test."
    else
        python3 scripts/test.py --tool ensirna --test-set data/ensirna/test.jsonl --ckpt "${CKPTS[@]}" \
            --save-dir results/ensirna --run-id ensirna \
            --output-csv results/ensirna/preds.csv --metrics-json results/ensirna/metrics.json
        if [ -n "${LEFTOUT_CSV}" ]; then
            python3 scripts/test.py --tool ensirna --test-set data/ensirna/leftout.jsonl --ckpt "${CKPTS[@]}" \
                --save-dir results/ensirna --run-id ensirna_leftout \
                --output-csv results/ensirna/preds_leftout.csv --metrics-json results/ensirna/metrics_leftout.json
        fi
    fi
fi

echo "=== DONE ==="
echo "Results in results/*/"
