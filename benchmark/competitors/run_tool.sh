#!/bin/bash
set -euo pipefail

ORIG_PWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="${REPO_ROOT}/benchmark"
DATA_ROOT="${SCRIPT_DIR}/data"
MODEL_ROOT="${SCRIPT_DIR}/models"
RESULTS_ROOT="${SCRIPT_DIR}/updated_validation_results"
TOOLS_ROOT="${SCRIPT_DIR}/tools"

export SIRBENCH_REPO_ROOT="${REPO_ROOT}"

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

maybe_prepare() {
    local label="$1"
    local out_path="$2"
    shift 2
    if [ -f "$out_path" ]; then
        echo "[$label] prepare skipped (exists)"
        return 0
    fi
    "$@"
}

maybe_train() {
    local label="$1"
    local model_path="$2"
    shift 2
    if [ -f "$model_path" ]; then
        echo "[$label] train skipped (model exists)"
        return 0
    fi
    "$@"
}

maybe_test() {
    local label="$1"
    local metrics_path="$2"
    shift 2
    if [ -f "$metrics_path" ]; then
        echo "[$label] test skipped (metrics exist)"
        return 0
    fi
    "$@"
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
SEED="42"
DETERMINISTIC=0
USE_ORIGINAL=0

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
        --seed)
            shift
            if [ $# -eq 0 ] || [[ "$1" == --* ]]; then
                echo "Missing value for --seed"
                exit 1
            fi
            SEED="$1"
            shift
            ;;
        --deterministic)
            DETERMINISTIC=1
            shift
            ;;
        --original)
            USE_ORIGINAL=1
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

make_abs() {
    local path="$1"
    if [ -z "${path}" ]; then
        echo ""
        return 0
    fi
    if [[ "${path}" = /* ]]; then
        echo "${path}"
    elif [ -e "${REPO_ROOT}/${path}" ]; then
        echo "${REPO_ROOT}/${path}"
    elif [ -e "${ORIG_PWD}/${path}" ]; then
        echo "${ORIG_PWD}/${path}"
    else
        echo "${REPO_ROOT}/${path}"
    fi
}

TRAIN_CSV="$(make_abs "${TRAIN_CSV}")"
VAL_CSV="$(make_abs "${VAL_CSV}")"
TEST_CSV="$(make_abs "${TEST_CSV}")"
LEFTOUT_CSV="$(make_abs "${LEFTOUT_CSV}")"

DETERMINISTIC_ARGS=()
if [ "${DETERMINISTIC}" = "1" ]; then
    DETERMINISTIC_ARGS+=(--deterministic)
fi
ORIGINAL_ARGS=()
if [ "${USE_ORIGINAL}" = "1" ]; then
    ORIGINAL_ARGS+=(--original-params)
    RESULTS_ROOT="${SCRIPT_DIR}/original_results"
fi

# ============ OLIGOFORMER ============
if [ "$RUN_OLIGOFORMER" = "1" ]; then
    echo "=== OLIGOFORMER ==="
    ensure_dirs oligoformer "${DATA_ROOT}/oligoformer" "${MODEL_ROOT}/oligoformer" "${RESULTS_ROOT}/oligoformer"

    RNAFM_ROOT="${TOOLS_ROOT}/oligoformer/oligoformer_src/RNA-FM"

    maybe_prepare "oligoformer" "${DATA_ROOT}/oligoformer/train.csv" python3 scripts/prepare.py --tool oligoformer --input-csv "$TRAIN_CSV" --output-dir "${DATA_ROOT}/oligoformer" --dataset-name train --run-rnafm --rnafm-root "$RNAFM_ROOT"
    maybe_prepare "oligoformer" "${DATA_ROOT}/oligoformer/val.csv" python3 scripts/prepare.py --tool oligoformer --input-csv "$VAL_CSV" --output-dir "${DATA_ROOT}/oligoformer" --dataset-name val --run-rnafm --rnafm-root "$RNAFM_ROOT"
    maybe_prepare "oligoformer" "${DATA_ROOT}/oligoformer/test.csv" python3 scripts/prepare.py --tool oligoformer --input-csv "$TEST_CSV" --output-dir "${DATA_ROOT}/oligoformer" --dataset-name test --run-rnafm --rnafm-root "$RNAFM_ROOT"
    if [ -n "${LEFTOUT_CSV}" ]; then
        maybe_prepare "oligoformer" "${DATA_ROOT}/oligoformer/leftout.csv" python3 scripts/prepare.py --tool oligoformer --input-csv "$LEFTOUT_CSV" --output-dir "${DATA_ROOT}/oligoformer" --dataset-name leftout --run-rnafm --rnafm-root "$RNAFM_ROOT"
    fi

    if [ "${USE_ORIGINAL}" = "1" ]; then
        maybe_train "oligoformer" "${MODEL_ROOT}/oligoformer/model.pt" python3 scripts/train.py --tool oligoformer --train-csv "${DATA_ROOT}/oligoformer/train.csv" --val-csv "${DATA_ROOT}/oligoformer/val.csv" --data-dir "${DATA_ROOT}/oligoformer" --model-dir "${MODEL_ROOT}/oligoformer" \
            --seed "${SEED}" "${DETERMINISTIC_ARGS[@]}" "${ORIGINAL_ARGS[@]}"
    else
        maybe_train "oligoformer" "${MODEL_ROOT}/oligoformer/model.pt" python3 scripts/train.py --tool oligoformer --train-csv "${DATA_ROOT}/oligoformer/train.csv" --val-csv "${DATA_ROOT}/oligoformer/val.csv" --data-dir "${DATA_ROOT}/oligoformer" --model-dir "${MODEL_ROOT}/oligoformer" \
            --epochs 100 --batch-size 16 --lr 1e-4 --weight-decay 0.999 --early-stopping 20 --early-stop-metric r2 --seed "${SEED}" "${DETERMINISTIC_ARGS[@]}"
    fi

    maybe_test "oligoformer" "${RESULTS_ROOT}/oligoformer/metrics.json" python3 scripts/test.py --tool oligoformer --test-csv "${DATA_ROOT}/oligoformer/test.csv" --data-dir "${DATA_ROOT}/oligoformer" --model-path "${MODEL_ROOT}/oligoformer/model.pt" --output-csv "${RESULTS_ROOT}/oligoformer/preds.csv" --metrics-json "${RESULTS_ROOT}/oligoformer/metrics.json"
    if [ -n "${LEFTOUT_CSV}" ]; then
        maybe_test "oligoformer" "${RESULTS_ROOT}/oligoformer/metrics_leftout.json" python3 scripts/test.py --tool oligoformer --test-csv "${DATA_ROOT}/oligoformer/leftout.csv" --data-dir "${DATA_ROOT}/oligoformer" --model-path "${MODEL_ROOT}/oligoformer/model.pt" --output-csv "${RESULTS_ROOT}/oligoformer/preds_leftout.csv" --metrics-json "${RESULTS_ROOT}/oligoformer/metrics_leftout.json"
    fi
fi

# ============ SIRNADISCOVERY ============
if [ "$RUN_SIRNADISCOVERY" = "1" ]; then
    echo "=== SIRNADISCOVERY ==="
    ensure_dirs sirnadiscovery "${DATA_ROOT}/sirnadiscovery" "${MODEL_ROOT}/sirnadiscovery" "${RESULTS_ROOT}/sirnadiscovery"

    PREPROCESS_TRAINVAL="${DATA_ROOT}/sirnadiscovery/siRNA_split_preprocess_trainval"
    PREPROCESS_TEST="${DATA_ROOT}/sirnadiscovery/siRNA_split_preprocess_test"
    PREPROCESS_LEFTOUT="${DATA_ROOT}/sirnadiscovery/siRNA_split_preprocess_leftout"
    RNA_AGO2_DIR="${DATA_ROOT}/sirnadiscovery/RNA_AGO2"
    RNA_AGO2_FALLBACK="${REPO_ROOT}/data/sirnadiscovery/RNA_AGO2"
    if [ ! -d "${RNA_AGO2_DIR}" ] && [ -d "${RNA_AGO2_FALLBACK}" ]; then
        RNA_AGO2_DIR="${RNA_AGO2_FALLBACK}"
    fi
    if [ ! -d "${RNA_AGO2_DIR}" ]; then
        echo "[sirnadiscovery] missing RNA_AGO2 features at ${RNA_AGO2_DIR}"
        echo "Provide RPISeq outputs per benchmark/competitors/tools/sirnadiscovery/README.md, then retry."
        exit 1
    fi

    if [ ! -f "${DATA_ROOT}/sirnadiscovery/trainval_input.csv" ]; then
python3 - <<PY
import csv

paths = [
    "${TRAIN_CSV}",
    "${VAL_CSV}",
]
out_path = "${DATA_ROOT}/sirnadiscovery/trainval_input.csv"

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
    fi
    if [ ! -d "${PREPROCESS_TRAINVAL}" ]; then
        python3 scripts/prepare.py --tool sirnadiscovery --input-csv "${DATA_ROOT}/sirnadiscovery/trainval_input.csv" --output-dir "${DATA_ROOT}/sirnadiscovery" --dataset-name trainval --run-preprocess --mrna-col extended_mRNA
        mv "${DATA_ROOT}/sirnadiscovery/siRNA_split_preprocess" "${PREPROCESS_TRAINVAL}"
    else
        echo "[sirnadiscovery] preprocess skipped (trainval exists)"
    fi
    if [ ! -d "${PREPROCESS_TEST}" ]; then
        python3 scripts/prepare.py --tool sirnadiscovery --input-csv "$TEST_CSV" --output-dir "${DATA_ROOT}/sirnadiscovery" --dataset-name test_pre --run-preprocess --mrna-col extended_mRNA
        mv "${DATA_ROOT}/sirnadiscovery/siRNA_split_preprocess" "${PREPROCESS_TEST}"
    fi
    if [ -n "${LEFTOUT_CSV}" ] && [ ! -d "${PREPROCESS_LEFTOUT}" ]; then
        python3 scripts/prepare.py --tool sirnadiscovery --input-csv "$LEFTOUT_CSV" --output-dir "${DATA_ROOT}/sirnadiscovery" --dataset-name leftout_pre --run-preprocess --mrna-col extended_mRNA
        mv "${DATA_ROOT}/sirnadiscovery/siRNA_split_preprocess" "${PREPROCESS_LEFTOUT}"
    fi

    maybe_prepare "sirnadiscovery" "${DATA_ROOT}/sirnadiscovery/train.csv" python3 scripts/prepare.py --tool sirnadiscovery --input-csv "$TRAIN_CSV" --output-dir "${DATA_ROOT}/sirnadiscovery" --dataset-name train --rna-ago2-dir "$RNA_AGO2_DIR" --mrna-col extended_mRNA
    maybe_prepare "sirnadiscovery" "${DATA_ROOT}/sirnadiscovery/val.csv" python3 scripts/prepare.py --tool sirnadiscovery --input-csv "$VAL_CSV" --output-dir "${DATA_ROOT}/sirnadiscovery" --dataset-name val --rna-ago2-dir "$RNA_AGO2_DIR" --mrna-col extended_mRNA
    maybe_prepare "sirnadiscovery" "${DATA_ROOT}/sirnadiscovery/test.csv" python3 scripts/prepare.py --tool sirnadiscovery --input-csv "$TEST_CSV" --output-dir "${DATA_ROOT}/sirnadiscovery" --dataset-name test --rna-ago2-dir "$RNA_AGO2_DIR" --mrna-col extended_mRNA
    if [ -n "${LEFTOUT_CSV}" ]; then
        maybe_prepare "sirnadiscovery" "${DATA_ROOT}/sirnadiscovery/leftout.csv" python3 scripts/prepare.py --tool sirnadiscovery --input-csv "$LEFTOUT_CSV" --output-dir "${DATA_ROOT}/sirnadiscovery" --dataset-name leftout --rna-ago2-dir "$RNA_AGO2_DIR" --mrna-col extended_mRNA
    fi

    if [ "${USE_ORIGINAL}" = "1" ]; then
        maybe_train "sirnadiscovery" "${MODEL_ROOT}/sirnadiscovery/model.keras" python3 scripts/train.py --tool sirnadiscovery --train-csv "${DATA_ROOT}/sirnadiscovery/train.csv" --val-csv "${DATA_ROOT}/sirnadiscovery/val.csv" --preprocess-dir "$PREPROCESS_TRAINVAL" --rna-ago2-dir "$RNA_AGO2_DIR" --model-dir "${MODEL_ROOT}/sirnadiscovery" \
            --seed "${SEED}" "${DETERMINISTIC_ARGS[@]}" "${ORIGINAL_ARGS[@]}"
    else
        maybe_train "sirnadiscovery" "${MODEL_ROOT}/sirnadiscovery/model.keras" python3 scripts/train.py --tool sirnadiscovery --train-csv "${DATA_ROOT}/sirnadiscovery/train.csv" --val-csv "${DATA_ROOT}/sirnadiscovery/val.csv" --preprocess-dir "$PREPROCESS_TRAINVAL" --rna-ago2-dir "$RNA_AGO2_DIR" --model-dir "${MODEL_ROOT}/sirnadiscovery" \
            --epochs 100 --early-stopping 20 --early-stop-metric val_r2_metric --seed "${SEED}" "${DETERMINISTIC_ARGS[@]}"
    fi

    maybe_test "sirnadiscovery" "${RESULTS_ROOT}/sirnadiscovery/metrics.json" python3 scripts/test.py --tool sirnadiscovery --test-csv "${DATA_ROOT}/sirnadiscovery/test.csv" --preprocess-dir "$PREPROCESS_TEST" --rna-ago2-dir "$RNA_AGO2_DIR" --model-path "${MODEL_ROOT}/sirnadiscovery/model.keras" --output-csv "${RESULTS_ROOT}/sirnadiscovery/preds.csv" --metrics-json "${RESULTS_ROOT}/sirnadiscovery/metrics.json"
    if [ -n "${LEFTOUT_CSV}" ]; then
        maybe_test "sirnadiscovery" "${RESULTS_ROOT}/sirnadiscovery/metrics_leftout.json" python3 scripts/test.py --tool sirnadiscovery --test-csv "${DATA_ROOT}/sirnadiscovery/leftout.csv" --preprocess-dir "$PREPROCESS_LEFTOUT" --rna-ago2-dir "$RNA_AGO2_DIR" --model-path "${MODEL_ROOT}/sirnadiscovery/model.keras" --output-csv "${RESULTS_ROOT}/sirnadiscovery/preds_leftout.csv" --metrics-json "${RESULTS_ROOT}/sirnadiscovery/metrics_leftout.json"
    fi
fi

# ============ SIRNABERT ============
if [ "$RUN_SIRNABERT" = "1" ]; then
    echo "=== SIRNABERT ==="
    ensure_dirs sirnabert "${DATA_ROOT}/sirnabert" "${MODEL_ROOT}/sirnabert" "${RESULTS_ROOT}/sirnabert"

    maybe_prepare "sirnabert" "${DATA_ROOT}/sirnabert/train.csv" python3 scripts/prepare.py --tool sirnabert --input-csv "$TRAIN_CSV" --output-dir "${DATA_ROOT}/sirnabert" --dataset-name train
    maybe_prepare "sirnabert" "${DATA_ROOT}/sirnabert/val.csv" python3 scripts/prepare.py --tool sirnabert --input-csv "$VAL_CSV" --output-dir "${DATA_ROOT}/sirnabert" --dataset-name val
    maybe_prepare "sirnabert" "${DATA_ROOT}/sirnabert/test.csv" python3 scripts/prepare.py --tool sirnabert --input-csv "$TEST_CSV" --output-dir "${DATA_ROOT}/sirnabert" --dataset-name test
    if [ -n "${LEFTOUT_CSV}" ]; then
        maybe_prepare "sirnabert" "${DATA_ROOT}/sirnabert/leftout.csv" python3 scripts/prepare.py --tool sirnabert --input-csv "$LEFTOUT_CSV" --output-dir "${DATA_ROOT}/sirnabert" --dataset-name leftout
    fi

    if [ "${USE_ORIGINAL}" = "1" ]; then
        maybe_train "sirnabert" "${MODEL_ROOT}/sirnabert/model.pt" python3 scripts/train.py --tool sirnabert --train-csv "${DATA_ROOT}/sirnabert/train.csv" --val-csv "${DATA_ROOT}/sirnabert/val.csv" --model-dir "${MODEL_ROOT}/sirnabert" \
            --seed "${SEED}" "${DETERMINISTIC_ARGS[@]}" "${ORIGINAL_ARGS[@]}"
    else
        maybe_train "sirnabert" "${MODEL_ROOT}/sirnabert/model.pt" python3 scripts/train.py --tool sirnabert --train-csv "${DATA_ROOT}/sirnabert/train.csv" --val-csv "${DATA_ROOT}/sirnabert/val.csv" --model-dir "${MODEL_ROOT}/sirnabert" \
            --epochs 100 --batch-size 100 --lr 5e-5 --max-len 16 --early-stopping 20 --early-stop-metric val_r2 --seed "${SEED}" "${DETERMINISTIC_ARGS[@]}"
    fi

    maybe_test "sirnabert" "${RESULTS_ROOT}/sirnabert/metrics.json" python3 scripts/test.py --tool sirnabert --test-csv "${DATA_ROOT}/sirnabert/test.csv" --model-path "${MODEL_ROOT}/sirnabert/model.pt" --output-csv "${RESULTS_ROOT}/sirnabert/preds.csv" --metrics-json "${RESULTS_ROOT}/sirnabert/metrics.json"
    if [ -n "${LEFTOUT_CSV}" ]; then
        maybe_test "sirnabert" "${RESULTS_ROOT}/sirnabert/metrics_leftout.json" python3 scripts/test.py --tool sirnabert --test-csv "${DATA_ROOT}/sirnabert/leftout.csv" --model-path "${MODEL_ROOT}/sirnabert/model.pt" --output-csv "${RESULTS_ROOT}/sirnabert/preds_leftout.csv" --metrics-json "${RESULTS_ROOT}/sirnabert/metrics_leftout.json"
    fi
fi

# ============ ATTSIOFF ============
if [ "$RUN_ATTSIOFF" = "1" ]; then
    echo "=== ATTSIOFF ==="
    ensure_dirs attsioff "${DATA_ROOT}/attsioff" "${MODEL_ROOT}/attsioff" "${RESULTS_ROOT}/attsioff"

    maybe_prepare "attsioff" "${DATA_ROOT}/attsioff/train.csv" python3 scripts/prepare.py --tool attsioff --input-csv "$TRAIN_CSV" --output-dir "${DATA_ROOT}/attsioff" --dataset-name train
    maybe_prepare "attsioff" "${DATA_ROOT}/attsioff/val.csv" python3 scripts/prepare.py --tool attsioff --input-csv "$VAL_CSV" --output-dir "${DATA_ROOT}/attsioff" --dataset-name val
    maybe_prepare "attsioff" "${DATA_ROOT}/attsioff/test.csv" python3 scripts/prepare.py --tool attsioff --input-csv "$TEST_CSV" --output-dir "${DATA_ROOT}/attsioff" --dataset-name test
    if [ -n "${LEFTOUT_CSV}" ]; then
        maybe_prepare "attsioff" "${DATA_ROOT}/attsioff/leftout.csv" python3 scripts/prepare.py --tool attsioff --input-csv "$LEFTOUT_CSV" --output-dir "${DATA_ROOT}/attsioff" --dataset-name leftout
    fi

    if [ "${USE_ORIGINAL}" = "1" ]; then
        maybe_train "attsioff" "${MODEL_ROOT}/attsioff/model.pt" python3 scripts/train.py --tool attsioff --train-csv "${DATA_ROOT}/attsioff/train.csv" --val-csv "${DATA_ROOT}/attsioff/val.csv" --data-dir "${DATA_ROOT}/attsioff" --model-dir "${MODEL_ROOT}/attsioff" \
            --seed "${SEED}" "${DETERMINISTIC_ARGS[@]}" "${ORIGINAL_ARGS[@]}"
    else
        maybe_train "attsioff" "${MODEL_ROOT}/attsioff/model.pt" python3 scripts/train.py --tool attsioff --train-csv "${DATA_ROOT}/attsioff/train.csv" --val-csv "${DATA_ROOT}/attsioff/val.csv" --data-dir "${DATA_ROOT}/attsioff" --model-dir "${MODEL_ROOT}/attsioff" \
            --batch-size 128 --epochs 100 --lr 0.005 --early-stopping 20 --early-stop-metric r2 --seed "${SEED}" "${DETERMINISTIC_ARGS[@]}"
    fi

    maybe_test "attsioff" "${RESULTS_ROOT}/attsioff/metrics.json" python3 scripts/test.py --tool attsioff --test-csv "${DATA_ROOT}/attsioff/test.csv" --data-dir "${DATA_ROOT}/attsioff" --model-path "${MODEL_ROOT}/attsioff/model.pt" --output-csv "${RESULTS_ROOT}/attsioff/preds.csv" --metrics-json "${RESULTS_ROOT}/attsioff/metrics.json"
    if [ -n "${LEFTOUT_CSV}" ]; then
        maybe_test "attsioff" "${RESULTS_ROOT}/attsioff/metrics_leftout.json" python3 scripts/test.py --tool attsioff --test-csv "${DATA_ROOT}/attsioff/leftout.csv" --data-dir "${DATA_ROOT}/attsioff" --model-path "${MODEL_ROOT}/attsioff/model.pt" --output-csv "${RESULTS_ROOT}/attsioff/preds_leftout.csv" --metrics-json "${RESULTS_ROOT}/attsioff/metrics_leftout.json"
    fi
fi

# ============ GNN4SIRNA ============
if [ "$RUN_GNN4SIRNA" = "1" ]; then
    echo "=== GNN4SIRNA ==="
    ensure_dirs gnn4sirna "${DATA_ROOT}/gnn4sirna" "${MODEL_ROOT}/gnn4sirna" "${RESULTS_ROOT}/gnn4sirna"

    maybe_prepare "gnn4sirna" "${DATA_ROOT}/gnn4sirna/train.csv" python3 scripts/prepare.py --tool gnn4sirna --input-csv "$TRAIN_CSV" --output-dir "${DATA_ROOT}/gnn4sirna" --dataset-name train --mrna-col extended_mRNA
    maybe_prepare "gnn4sirna" "${DATA_ROOT}/gnn4sirna/val.csv" python3 scripts/prepare.py --tool gnn4sirna --input-csv "$VAL_CSV" --output-dir "${DATA_ROOT}/gnn4sirna" --dataset-name val --mrna-col extended_mRNA
    maybe_prepare "gnn4sirna" "${DATA_ROOT}/gnn4sirna/test.csv" python3 scripts/prepare.py --tool gnn4sirna --input-csv "$TEST_CSV" --output-dir "${DATA_ROOT}/gnn4sirna" --dataset-name test --mrna-col extended_mRNA
    if [ -n "${LEFTOUT_CSV}" ]; then
        maybe_prepare "gnn4sirna" "${DATA_ROOT}/gnn4sirna/leftout.csv" python3 scripts/prepare.py --tool gnn4sirna --input-csv "$LEFTOUT_CSV" --output-dir "${DATA_ROOT}/gnn4sirna" --dataset-name leftout --mrna-col extended_mRNA
    fi
    if [ ! -f "${DATA_ROOT}/gnn4sirna/trainval_input.csv" ]; then
python3 - <<PY
import csv

paths = [
    "${TRAIN_CSV}",
    "${VAL_CSV}",
]
out_path = "${DATA_ROOT}/gnn4sirna/trainval_input.csv"

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
    fi
    if [ ! -d "${DATA_ROOT}/gnn4sirna/processed/trainval" ]; then
        python3 scripts/prepare.py --tool gnn4sirna --input-csv "${DATA_ROOT}/gnn4sirna/trainval_input.csv" --output-dir "${DATA_ROOT}/gnn4sirna" --dataset-name trainval --mrna-col extended_mRNA
    else
        echo "[gnn4sirna] prepare skipped (processed/trainval exists)"
    fi
    if [ ! -d "${DATA_ROOT}/gnn4sirna/processed/test" ]; then
        python3 scripts/prepare.py --tool gnn4sirna --input-csv "$TEST_CSV" --output-dir "${DATA_ROOT}/gnn4sirna" --dataset-name test --mrna-col extended_mRNA
    fi
    if [ -n "${LEFTOUT_CSV}" ] && [ ! -d "${DATA_ROOT}/gnn4sirna/processed/leftout" ]; then
        python3 scripts/prepare.py --tool gnn4sirna --input-csv "$LEFTOUT_CSV" --output-dir "${DATA_ROOT}/gnn4sirna" --dataset-name leftout --mrna-col extended_mRNA
    fi

    if [ "${USE_ORIGINAL}" = "1" ]; then
        maybe_train "gnn4sirna" "${MODEL_ROOT}/gnn4sirna/model.keras" python3 scripts/train.py --tool gnn4sirna --train-csv "${DATA_ROOT}/gnn4sirna/train.csv" --val-csv "${DATA_ROOT}/gnn4sirna/val.csv" --processed-dir "${DATA_ROOT}/gnn4sirna/processed/trainval" --model-dir "${MODEL_ROOT}/gnn4sirna" \
            --seed "${SEED}" "${DETERMINISTIC_ARGS[@]}" "${ORIGINAL_ARGS[@]}"
    else
        maybe_train "gnn4sirna" "${MODEL_ROOT}/gnn4sirna/model.keras" python3 scripts/train.py --tool gnn4sirna --train-csv "${DATA_ROOT}/gnn4sirna/train.csv" --val-csv "${DATA_ROOT}/gnn4sirna/val.csv" --processed-dir "${DATA_ROOT}/gnn4sirna/processed/trainval" --model-dir "${MODEL_ROOT}/gnn4sirna" \
            --batch-size 60 --epochs 100 --lr 1e-3 --loss mse --early-stopping 20 --early-stop-metric val_r2_metric --seed "${SEED}" "${DETERMINISTIC_ARGS[@]}"
    fi

    maybe_test "gnn4sirna" "${RESULTS_ROOT}/gnn4sirna/metrics.json" python3 scripts/test.py --tool gnn4sirna --test-csv "${DATA_ROOT}/gnn4sirna/test.csv" --processed-dir "${DATA_ROOT}/gnn4sirna/processed/test" --model-path "${MODEL_ROOT}/gnn4sirna/model.keras" --output-csv "${RESULTS_ROOT}/gnn4sirna/preds.csv" --metrics-json "${RESULTS_ROOT}/gnn4sirna/metrics.json"
    if [ -n "${LEFTOUT_CSV}" ]; then
        maybe_test "gnn4sirna" "${RESULTS_ROOT}/gnn4sirna/metrics_leftout.json" python3 scripts/test.py --tool gnn4sirna --test-csv "${DATA_ROOT}/gnn4sirna/leftout.csv" --processed-dir "${DATA_ROOT}/gnn4sirna/processed/leftout" --model-path "${MODEL_ROOT}/gnn4sirna/model.keras" --output-csv "${RESULTS_ROOT}/gnn4sirna/preds_leftout.csv" --metrics-json "${RESULTS_ROOT}/gnn4sirna/metrics_leftout.json"
    fi
fi

# ============ ENSIRNA ============
if [ "$RUN_ENSIRNA" = "1" ]; then
    echo "=== ENSIRNA ==="
    ensure_dirs ensirna "${DATA_ROOT}/ensirna" "${MODEL_ROOT}/ensirna" "${RESULTS_ROOT}/ensirna"

    maybe_prepare "ensirna" "${DATA_ROOT}/ensirna/train.jsonl" python3 scripts/prepare.py --tool ensirna --input-csv "$TRAIN_CSV" --output-jsonl "${DATA_ROOT}/ensirna/train.jsonl"
    maybe_prepare "ensirna" "${DATA_ROOT}/ensirna/val.jsonl" python3 scripts/prepare.py --tool ensirna --input-csv "$VAL_CSV" --output-jsonl "${DATA_ROOT}/ensirna/val.jsonl"
    maybe_prepare "ensirna" "${DATA_ROOT}/ensirna/test.jsonl" python3 scripts/prepare.py --tool ensirna --input-csv "$TEST_CSV" --output-jsonl "${DATA_ROOT}/ensirna/test.jsonl"
    if [ -n "${LEFTOUT_CSV}" ]; then
        maybe_prepare "ensirna" "${DATA_ROOT}/ensirna/leftout.jsonl" python3 scripts/prepare.py --tool ensirna --input-csv "$LEFTOUT_CSV" --output-jsonl "${DATA_ROOT}/ensirna/leftout.jsonl"
    fi

    shopt -s globstar nullglob
    CKPTS=(${MODEL_ROOT}/ensirna/**/*.ckpt)
    shopt -u globstar nullglob
    if [ ${#CKPTS[@]} -eq 0 ]; then
        if [ "${USE_ORIGINAL}" = "1" ]; then
            python3 scripts/train.py --tool ensirna --train-set "${DATA_ROOT}/ensirna/train.jsonl" --valid-set "${DATA_ROOT}/ensirna/val.jsonl" --model-dir "${MODEL_ROOT}/ensirna" \
                --batch-size 16 --seed "${SEED}" "${ORIGINAL_ARGS[@]}"
        else
            python3 scripts/train.py --tool ensirna --train-set "${DATA_ROOT}/ensirna/train.jsonl" --valid-set "${DATA_ROOT}/ensirna/val.jsonl" --model-dir "${MODEL_ROOT}/ensirna" \
                --batch-size 16 --lr 1e-4 --final_lr 1e-5 --max_epoch 100 --patience 20 --val_metric r2 --seed "${SEED}"
        fi
    else
        echo "[ensirna] train skipped (checkpoints exist)"
    fi

    shopt -s globstar nullglob
    CKPTS=(${MODEL_ROOT}/ensirna/**/*.ckpt)
    shopt -u globstar nullglob
    if [ ${#CKPTS[@]} -eq 0 ]; then
        echo "No ENsiRNA checkpoints found in ${MODEL_ROOT}/ensirna; skipping test."
    else
        maybe_test "ensirna" "${RESULTS_ROOT}/ensirna/metrics.json" python3 scripts/test.py --tool ensirna --test-set "${DATA_ROOT}/ensirna/test.jsonl" --ckpt "${CKPTS[@]}" \
            --save-dir "${RESULTS_ROOT}/ensirna" --run-id ensirna \
            --output-csv "${RESULTS_ROOT}/ensirna/preds.csv" --metrics-json "${RESULTS_ROOT}/ensirna/metrics.json"
        if [ -n "${LEFTOUT_CSV}" ]; then
            maybe_test "ensirna" "${RESULTS_ROOT}/ensirna/metrics_leftout.json" python3 scripts/test.py --tool ensirna --test-set "${DATA_ROOT}/ensirna/leftout.jsonl" --ckpt "${CKPTS[@]}" \
                --save-dir "${RESULTS_ROOT}/ensirna" --run-id ensirna_leftout \
                --output-csv "${RESULTS_ROOT}/ensirna/preds_leftout.csv" --metrics-json "${RESULTS_ROOT}/ensirna/metrics_leftout.json"
        fi
    fi
fi

echo "=== DONE ==="
echo "Results in ${RESULTS_ROOT}/*/"
