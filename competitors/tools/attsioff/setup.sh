#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$*" == *"--docker"* ]]; then
    image_tag="${IMAGE_TAG:-attsioff:latest}"

    CKPT_DIR="checkpoints"
    CKPT_FILE="${CKPT_DIR}/RNA-FM_pretrained.pth"
    CKPT_URL="https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=RNA-FM_pretrained.pth"
    EXPECTED_SIZE=1194424423
    mkdir -p "${CKPT_DIR}"
    size=0
    if [ -f "${CKPT_FILE}" ]; then
        size=$(wc -c < "${CKPT_FILE}")
    fi
    if [ "${size}" -ne "${EXPECTED_SIZE}" ]; then
        curl -L -C - -o "${CKPT_FILE}" "${CKPT_URL}"
    fi
    if [ ! -f "${CKPT_FILE}" ]; then
        echo "Checkpoint download failed: ${CKPT_FILE}"
        exit 1
    fi
    size=$(wc -c < "${CKPT_FILE}")
    if [ "${size}" -ne "${EXPECTED_SIZE}" ]; then
        echo "Checkpoint size mismatch: ${size} (expected ${EXPECTED_SIZE})."
        echo "Rerun setup to resume the download."
        exit 1
    fi

    BUILD_ARGS=""
    [ -n "$http_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg http_proxy=$http_proxy"
    [ -n "$https_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg https_proxy=$https_proxy"
    [ -n "$HTTP_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTP_PROXY=$HTTP_PROXY"
    [ -n "$HTTPS_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTPS_PROXY=$HTTPS_PROXY"

    docker build --platform linux/amd64 $BUILD_ARGS -t "$image_tag" .
fi

if [ ! -d "attsioff_src" ]; then
    git clone https://github.com/2333liubin/AttSiOff.git attsioff_src
fi

if [ ! -d "attsioff_src/RNA-FM" ]; then
    git clone https://github.com/ml4bio/RNA-FM.git attsioff_src/RNA-FM
fi
