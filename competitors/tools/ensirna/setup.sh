#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$*" == *"--docker"* ]]; then
    image_tag="${IMAGE_TAG:-ensirna:latest}"

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
