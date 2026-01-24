#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

OUT_DIR="${ROSETTA_OUT_DIR:-$(pwd)/rosetta}"
DEFAULT_URL="https://downloads.rosettacommons.org/downloads/academic/3.15/rosetta_binary_ubuntu_3.15_bundle.tar.bz2"
SOURCE="${ROSETTA_TARBALL:-$DEFAULT_URL}"

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

tar_extract_stdin() {
    tar -xj -f - -C "${TMP_DIR}" --wildcards --wildcards-match-slash \
        "*/database/**" \
        "*/main/source/bin/rna_denovo*linux*release" \
        "*/main/tools/rna_tools/**/extract_lowscore_decoys.py"
}

tar_extract_file() {
    tar -xjf "${SOURCE}" -C "${TMP_DIR}" --wildcards --wildcards-match-slash \
        "*/database/**" \
        "*/main/source/bin/rna_denovo*linux*release" \
        "*/main/tools/rna_tools/**/extract_lowscore_decoys.py"
}

if [[ "${SOURCE}" =~ ^https?:// ]]; then
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required to download Rosetta."
        exit 1
    fi
    echo "Downloading and extracting Rosetta bundle from ${SOURCE}..."
    if ! curl -L --fail "${SOURCE}" | tar_extract_stdin; then
        echo "Download or extraction failed: ${SOURCE}"
        exit 1
    fi
else
    if [ ! -f "${SOURCE}" ]; then
        echo "Rosetta bundle not found: ${SOURCE}"
        exit 1
    fi
    echo "Extracting Rosetta bundle from ${SOURCE}..."
    tar_extract_file
fi

BIN_PATH=$(find "${TMP_DIR}" -path "*/main/source/bin/rna_denovo*linux*release" -print -quit)
if [ -z "${BIN_PATH}" ]; then
    BIN_PATH=$(find "${TMP_DIR}" -path "*/bin/rna_denovo.default.linuxgccrelease" -print -quit)
fi
if [ -z "${BIN_PATH}" ]; then
    echo "Could not locate rna_denovo in bundle."
    exit 1
fi

BIN_ROOT="${BIN_PATH%/main/source/bin/*}"
if [ "${BIN_ROOT}" = "${BIN_PATH}" ]; then
    BIN_ROOT="${BIN_PATH%/bin/*}"
fi

DB_PATH=$(find "${TMP_DIR}" -path "*/database/chemical/residue_type_sets/fa_standard/residue_types.txt" -print -quit)
if [ -z "${DB_PATH}" ]; then
    echo "Rosetta database not found in bundle."
    exit 1
fi
DB_ROOT="${DB_PATH%/database/chemical/residue_type_sets/fa_standard/residue_types.txt}"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

echo "Copying Rosetta binaries from ${BIN_ROOT}..."
cp -a "${BIN_ROOT}/." "${OUT_DIR}/"

if [ "${DB_ROOT}" != "${BIN_ROOT}" ]; then
    echo "Copying Rosetta database from ${DB_ROOT}..."
    if [ -d "${DB_ROOT}/database" ]; then
        cp -a "${DB_ROOT}/database" "${OUT_DIR}/"
    else
        echo "Database directory not found under ${DB_ROOT}."
        exit 1
    fi
fi

if [ ! -f "${OUT_DIR}/database/chemical/residue_type_sets/fa_standard/residue_types.txt" ]; then
    echo "Rosetta database missing under ${OUT_DIR}/database."
    exit 1
fi

echo "Rosetta extracted to ${OUT_DIR}"
