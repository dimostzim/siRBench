#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$*" == *"--docker"* ]]; then
    if ! docker image inspect oligoformer:latest >/dev/null 2>&1; then
        echo "Building oligoformer first (needed for RNA-FM weights)..."
        ../../setup.sh --tool oligoformer
    fi

    image_tag="${IMAGE_TAG:-ensirna:latest}"

    BUILD_ARGS=""
    [ -n "$http_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg http_proxy=$http_proxy"
    [ -n "$https_proxy" ] && BUILD_ARGS="$BUILD_ARGS --build-arg https_proxy=$https_proxy"
    [ -n "$HTTP_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTP_PROXY=$HTTP_PROXY"
    [ -n "$HTTPS_PROXY" ] && BUILD_ARGS="$BUILD_ARGS --build-arg HTTPS_PROXY=$HTTPS_PROXY"

    ROSETTA_DIR="${ROSETTA_DIR:-$(pwd)/rosetta}"
    ROSETTA_DB_FILE="${ROSETTA_DIR}/database/chemical/residue_type_sets/fa_standard/residue_types.txt"
    if [ ! -f "${ROSETTA_DB_FILE}" ]; then
        if [ -x "./fetch_rosetta.sh" ]; then
            if [ -d "${ROSETTA_DIR}" ]; then
                echo "Rosetta database missing; re-downloading the default Rosetta bundle (with database)."
            else
                echo "Rosetta not found; downloading the default Rosetta bundle (with database)."
            fi
            ROSETTA_OUT_DIR="${ROSETTA_DIR}" ./fetch_rosetta.sh
        else
            echo "fetch_rosetta.sh is not executable."
            exit 1
        fi
    fi

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
