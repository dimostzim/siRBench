#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$*" == *"--docker"* ]]; then
    image_tag="${IMAGE_TAG:-gnn4sirna:latest}"

    BUILD_ARGS=""
    [ -n "$http_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg http_proxy=$http_proxy"
    [ -n "$https_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg https_proxy=$https_proxy"
    [ -n "$HTTP_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTP_PROXY=$HTTP_PROXY"
    [ -n "$HTTPS_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTPS_PROXY=$HTTPS_PROXY"

    docker build --platform linux/amd64 $BUILD_ARGS -t "$image_tag" .
fi

if [ ! -d "gnn4sirna_src" ]; then
    git clone --no-checkout https://github.com/BCB4PM/GNN4siRNA.git gnn4sirna_src
    git -C "gnn4sirna_src" checkout --detach 5247663c6eb3a4939f1eb7f385f9be91d6324d60
fi

if [ "$(git -C "gnn4sirna_src" rev-parse HEAD)" != "5247663c6eb3a4939f1eb7f385f9be91d6324d60" ]; then
    echo "Expected gnn4sirna_src at 5247663c6eb3a4939f1eb7f385f9be91d6324d60; use a separate checkout for another version." >&2
    exit 1
fi
