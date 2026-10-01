#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$*" == *"--docker"* ]]; then
    image_tag="${IMAGE_TAG:-attsioff:latest}"

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

    BUILD_ARGS=""
    [ -n "$http_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg http_proxy=$http_proxy"
    [ -n "$https_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg https_proxy=$https_proxy"
    [ -n "$HTTP_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTP_PROXY=$HTTP_PROXY"
    [ -n "$HTTPS_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTPS_PROXY=$HTTPS_PROXY"

    docker build --platform linux/amd64 $BUILD_ARGS -t "$image_tag" .
fi

if [ ! -d "attsioff_src" ]; then
    git clone --no-checkout https://github.com/2333liubin/AttSiOff.git attsioff_src
    git -C "attsioff_src" checkout --detach f1d8ed6fdd93cc2708d910a411c26d3f3fdf08db
fi

if [ ! -d "attsioff_src/RNA-FM" ]; then
    git clone --no-checkout https://github.com/ml4bio/RNA-FM.git attsioff_src/RNA-FM
    git -C "attsioff_src/RNA-FM" checkout --detach 348951516e0963d22bbb33b3c9fc18c89081d38e
fi

if [ "$(git -C "attsioff_src" rev-parse HEAD)" != "f1d8ed6fdd93cc2708d910a411c26d3f3fdf08db" ]; then
    echo "Expected attsioff_src at f1d8ed6fdd93cc2708d910a411c26d3f3fdf08db; use a separate checkout for another version." >&2
    exit 1
fi

if [ "$(git -C "attsioff_src/RNA-FM" rev-parse HEAD)" != "348951516e0963d22bbb33b3c9fc18c89081d38e" ]; then
    echo "Expected attsioff_src/RNA-FM at 348951516e0963d22bbb33b3c9fc18c89081d38e; use a separate checkout for another version." >&2
    exit 1
fi
