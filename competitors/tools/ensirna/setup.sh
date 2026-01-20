#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$*" == *"--docker"* ]]; then
    image_tag="${IMAGE_TAG:-ensirna:latest}"

    BUILD_ARGS=""
    [ -n "$http_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg http_proxy=$http_proxy"
    [ -n "$https_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg https_proxy=$https_proxy"
    [ -n "$HTTP_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTP_PROXY=$HTTP_PROXY"
    [ -n "$HTTPS_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTPS_PROXY=$HTTPS_PROXY"

    docker build --platform linux/amd64 $BUILD_ARGS -t "$image_tag" .
fi

if [ ! -d "ensirna_src" ]; then
    git clone https://github.com/tanwenchong/ENsiRNA.git ensirna_src
fi

PATCH_FILE="$(pwd)/patches/get_pdb.py"
TARGET_FILE="$(pwd)/ensirna_src/ENsiRNA/data/get_pdb.py"
if [ -f "${PATCH_FILE}" ]; then
    cp "${PATCH_FILE}" "${TARGET_FILE}"
fi
