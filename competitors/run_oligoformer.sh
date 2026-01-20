#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/../data}"

TRAIN_CSV="${TRAIN_CSV:-${DATA_ROOT}/siRBench_full_base_1000_train.csv}"
VAL_CSV="${VAL_CSV:-${DATA_ROOT}/siRBench_full_base_1000_val.csv}"
TEST_CSV="${TEST_CSV:-${DATA_ROOT}/siRBench_full_base_1000_test.csv}"

OUT_DATA="${OUT_DATA:-${ROOT_DIR}/data/oligoformer}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/oligoformer}"
PREDS="${PREDS:-${ROOT_DIR}/preds_oligoformer.csv}"
METRICS_JSON="${METRICS_JSON:-${ROOT_DIR}/metrics_oligoformer.json}"

RNAFM_ROOT="${RNAFM_ROOT:-${ROOT_DIR}/tools/oligoformer/oligoformer_src/RNA-FM}"
USE_RNAFM="${USE_RNAFM:-1}"

SUDO=""
if ! docker info >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "Docker is not accessible and sudo is not available."
    exit 1
  fi
fi

run() {
  echo "+ $*"
  $SUDO "$@"
}

run "${ROOT_DIR}/setup.sh" --tool oligoformer

PREPARE_ARGS=(--tool oligoformer --output-dir "${OUT_DATA}")
if [ "${USE_RNAFM}" = "1" ]; then
  PREPARE_ARGS+=(--run-rnafm --rnafm-root "${RNAFM_ROOT}")
fi

run python3 "${ROOT_DIR}/prepare.py" "${PREPARE_ARGS[@]}" \
  --input-csv "${TRAIN_CSV}" --dataset-name train
run python3 "${ROOT_DIR}/prepare.py" "${PREPARE_ARGS[@]}" \
  --input-csv "${VAL_CSV}" --dataset-name val
run python3 "${ROOT_DIR}/prepare.py" "${PREPARE_ARGS[@]}" \
  --input-csv "${TEST_CSV}" --dataset-name test

run python3 "${ROOT_DIR}/train.py" --tool oligoformer \
  --train-csv "${OUT_DATA}/train.csv" \
  --val-csv "${OUT_DATA}/val.csv" \
  --data-dir "${OUT_DATA}" \
  --model-dir "${MODEL_DIR}"

run python3 "${ROOT_DIR}/test.py" --tool oligoformer \
  --test-csv "${OUT_DATA}/test.csv" \
  --data-dir "${OUT_DATA}" \
  --model-path "${MODEL_DIR}/model.pt" \
  --output-csv "${PREDS}" \
  --metrics-json "${METRICS_JSON}"

echo "Done. Predictions: ${PREDS}"
echo "Metrics: ${METRICS_JSON}"
