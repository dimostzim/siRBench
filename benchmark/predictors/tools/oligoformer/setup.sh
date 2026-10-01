#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$*" == *"--docker"* ]]; then
    image_tag="${IMAGE_TAG:-oligoformer:latest}"

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

if [ ! -d "oligoformer_src" ]; then
    git clone --no-checkout https://github.com/lulab/OligoFormer.git oligoformer_src
    git -C "oligoformer_src" checkout --detach e2f53ad63387bbe166bf123949151e2bc9bf6ec3
fi

if [ ! -d "oligoformer_src/RNA-FM" ]; then
    git clone --no-checkout https://github.com/ml4bio/RNA-FM.git oligoformer_src/RNA-FM
    git -C "oligoformer_src/RNA-FM" checkout --detach 348951516e0963d22bbb33b3c9fc18c89081d38e
fi

if [ "$(git -C "oligoformer_src" rev-parse HEAD)" != "e2f53ad63387bbe166bf123949151e2bc9bf6ec3" ]; then
    echo "Expected oligoformer_src at e2f53ad63387bbe166bf123949151e2bc9bf6ec3; use a separate checkout for another version." >&2
    exit 1
fi

if [ "$(git -C "oligoformer_src/RNA-FM" rev-parse HEAD)" != "348951516e0963d22bbb33b3c9fc18c89081d38e" ]; then
    echo "Expected oligoformer_src/RNA-FM at 348951516e0963d22bbb33b3c9fc18c89081d38e; use a separate checkout for another version." >&2
    exit 1
fi
