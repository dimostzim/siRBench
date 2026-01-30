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
    RNA_DENOVO="${ROSETTA_DIR}/main/source/bin/rna_denovo.static.linuxgccrelease"
    if [ ! -e "${RNA_DENOVO}" ]; then
        if [ -x "./fetch_rosetta.sh" ]; then
            echo "Rosetta rna_denovo missing; re-downloading the default Rosetta bundle."
            ROSETTA_OUT_DIR="${ROSETTA_DIR}" ./fetch_rosetta.sh
        else
            echo "fetch_rosetta.sh is not executable."
            exit 1
        fi
    fi
    EXTRACT_PDBS="$(find "${ROSETTA_DIR}/main/source/bin" -maxdepth 1 -name 'extract_pdbs*linux*release' -print -quit)"
    if [ -z "${EXTRACT_PDBS}" ]; then
        if [ -x "./fetch_rosetta.sh" ]; then
            echo "Rosetta extract_pdbs missing; re-downloading the default Rosetta bundle."
            ROSETTA_OUT_DIR="${ROSETTA_DIR}" ./fetch_rosetta.sh
        else
            echo "fetch_rosetta.sh is not executable."
            exit 1
        fi
    fi
    BIN_DIR="${ROSETTA_DIR}/main/source/bin"
    EXTRACT_STATIC="${BIN_DIR}/extract_pdbs.static.linuxgccrelease"
    EXTRACT_BIN="${BIN_DIR}/extract_pdbs.linuxgccrelease"
    if [ -f "${EXTRACT_STATIC}" ] && [ ! -e "${EXTRACT_BIN}" ]; then
        (cd "${BIN_DIR}" && ln -s extract_pdbs.static.linuxgccrelease extract_pdbs.linuxgccrelease)
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
