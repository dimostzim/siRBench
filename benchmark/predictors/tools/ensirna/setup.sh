#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$*" == *"--docker"* ]]; then
    image_tag="${IMAGE_TAG:-ensirna:latest}"

    CKPT_DIR="checkpoints"
    CKPT_FILE="${CKPT_DIR}/RNA-FM_pretrained.pth"
    CKPT_URL="https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth"
    EXPECTED_SIZE=1194424423
    mkdir -p "${CKPT_DIR}"
    wget -c -O "${CKPT_FILE}" "${CKPT_URL}"
    size=$(wc -c < "${CKPT_FILE}")
    if [ "${size}" -ne "${EXPECTED_SIZE}" ]; then
        echo "Checkpoint size mismatch: ${size} (expected ${EXPECTED_SIZE})."
        exit 1
    fi

    VIENNA_ARCHIVE="${CKPT_DIR}/ViennaRNA-2.6.4.tar.gz"
    if [ ! -f "${VIENNA_ARCHIVE}" ]; then
        wget -O "${VIENNA_ARCHIVE}" https://github.com/ViennaRNA/ViennaRNA/releases/download/v2.6.4/ViennaRNA-2.6.4.tar.gz
    fi
    echo "3a997a6aa6a3ce1af4898aa559acb053e820aa74bac06ef7726b9aa97a053788  ${VIENNA_ARCHIVE}" | sha256sum --check

    BUILD_ARGS=""
    [ -n "$http_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg http_proxy=$http_proxy"
    [ -n "$https_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg https_proxy=$https_proxy"
    [ -n "$HTTP_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTP_PROXY=$HTTP_PROXY"
    [ -n "$HTTPS_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTPS_PROXY=$HTTPS_PROXY"

    ROSETTA_DIR="${ROSETTA_DIR:-$(pwd)/rosetta}"
    ROSETTA_OUT_DIR="${ROSETTA_DIR}" ./fetch_rosetta.sh

    docker build --platform linux/amd64 $BUILD_ARGS -t "$image_tag" .
fi

if [ ! -d "ensirna_src" ]; then
    git clone --no-checkout https://github.com/tanwenchong/ENsiRNA.git ensirna_src
    git -C ensirna_src checkout --detach 028824341635903f3c661f5d1cc737de106493d5
fi

if [ "$(git -C ensirna_src rev-parse HEAD)" != "028824341635903f3c661f5d1cc737de106493d5" ]; then
    echo "ENsiRNA source must be pinned to 028824341635903f3c661f5d1cc737de106493d5." >&2
    exit 1
fi

PATCH_FILE="$(pwd)/patches/get_pdb.py"
TARGET_FILE="$(pwd)/ensirna_src/ENsiRNA/data/get_pdb.py"
if [ -f "${PATCH_FILE}" ]; then
    cp "${PATCH_FILE}" "${TARGET_FILE}"
fi

PATCH_FILE="$(pwd)/patches/dataset.py"
TARGET_FILE="$(pwd)/ensirna_src/ENsiRNA/data/dataset.py"
if [ -f "${PATCH_FILE}" ]; then
    cp "${PATCH_FILE}" "${TARGET_FILE}"
fi

PATCH_FILE="$(pwd)/patches/run.py"
TARGET_FILE="$(pwd)/ensirna_src/ENsiRNA/run.py"
if [ -f "${PATCH_FILE}" ]; then
    cp "${PATCH_FILE}" "${TARGET_FILE}"
fi

PATCH_FILE="$(pwd)/patches/train.py"
TARGET_FILE="$(pwd)/ensirna_src/ENsiRNA/train.py"
if [ -f "${PATCH_FILE}" ]; then
    cp "${PATCH_FILE}" "${TARGET_FILE}"
fi

PATCH_FILE="$(pwd)/patches/abs_trainer.py"
TARGET_FILE="$(pwd)/ensirna_src/ENsiRNA/trainer/abs_trainer.py"
if [ -f "${PATCH_FILE}" ]; then
    cp "${PATCH_FILE}" "${TARGET_FILE}"
fi

PATCH_FILE="$(pwd)/patches/random_seed.py"
TARGET_FILE="$(pwd)/ensirna_src/ENsiRNA/utils/random_seed.py"
if [ -f "${PATCH_FILE}" ]; then
    cp "${PATCH_FILE}" "${TARGET_FILE}"
fi

PATCH_FILE="$(pwd)/patches/RNAmaskModel_trainer.py"
TARGET_FILE="$(pwd)/ensirna_src/ENsiRNA/trainer/RNAmaskModel_trainer.py"
cp "${PATCH_FILE}" "${TARGET_FILE}"

cp patches/embedding_utils.py ensirna_src/ENsiRNA/data/embedding_utils.py
