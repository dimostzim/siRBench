#!/usr/bin/env bash
# Retrieve the exact Rosetta runtime shipped in the released ENsiRNA image.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out_dir=$(readlink -m "${ROSETTA_OUT_DIR:-$script_dir/rosetta}")
image_sha=1a9c8b80a2d5b5943770fb5e736264cb5234997181df98d7704bde673f09167f
layer_sha=376d3c0a0fd79225ec273944bee58e3711e346ba61f0293f415410c806e2323c
runtime_sha=a2008e7d09e25bf5d93d4432c826bf5723aecb9341556edba260503b9700217d
symlinks_sha=fe09ebc5764abd6ff2017136db6cc2c6924bb587b40f290416503935174654fe
license_sha=740846d9cfb0baa01baf0ac2f367a4ea047ef9a87bd00ea9e0b1d2e422dce048
rna_sha=85b022f1a4b813b7e4dbd3b8ca4649559078c9bbe6ea5680ca011939c845f7a6
extract_sha=a6e47a40664d08b761930d315359ceeeae42cd3bb56b15214a363ce6d9d206cd
archive_root=app/ENsiRNA-main/rosetta/rosetta.binary.linux.release-371
registry=https://registry-1.docker.io/v2/tanwenchong/ensirna
if [ -n "${ROSETTA_TARBALL:-}" ]; then
    echo 'ROSETTA_TARBALL is retired; use ROSETTA_LAYER_ARCHIVE for the pinned official image layer.' >&2
    exit 1
fi
mkdir -p "$(dirname "$out_dir")"
tmp_dir=$(mktemp -d "$(dirname "$out_dir")/.ensirna-rosetta.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT

check_sha() {
    local actual
    actual=$(sha256sum "$1" | cut -d ' ' -f 1)
    if [ "$actual" != "$2" ]; then
        echo "SHA256 mismatch for $1: $actual (expected $2)" >&2
        exit 1
    fi
}

registry_download() {
    if [ -z "${registry_token:-}" ]; then
        registry_token=$(curl --fail --silent --show-error 'https://auth.docker.io/token?service=registry.docker.io&scope=repository:tanwenchong/ensirna:pull' | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')
    fi
    curl --fail --location --silent --show-error \
        -H "Authorization: Bearer $registry_token" \
        -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
        "$registry/$1" -o "$2"
}

cp "${ROSETTA_IMAGE_MANIFEST:-$script_dir/rosetta-image-manifest.json}" "$tmp_dir/image-manifest.json"
check_sha "$tmp_dir/image-manifest.json" "$image_sha"
python3 - "$tmp_dir/image-manifest.json" "$layer_sha" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
assert manifest['config']['digest'] == 'sha256:b7a48dcc8fb2592c89d41f1c939dffd3a8a1bc67a5a6cb63fd48673d07ef20f0'
assert any(layer['digest'] == 'sha256:' + sys.argv[2] and layer['size'] == 2802379734 for layer in manifest['layers'])
PY

if [ -e "$out_dir" ]; then
    runtime=$out_dir
else
    layer=${ROSETTA_LAYER_ARCHIVE:-$tmp_dir/app-layer.tar.gz}
    if [ -z "${ROSETTA_LAYER_ARCHIVE:-}" ]; then
        registry_download "blobs/sha256:$layer_sha" "$layer"
    fi
    check_sha "$layer" "$layer_sha"
    mkdir "$tmp_dir/extracted"
    build_path=main/source/build/src/release/linux/5.4/64/x86/gcc/4.8/static
    tar --no-same-owner -xf "$layer" -C "$tmp_dir/extracted" \
        "$archive_root/main/database" "$archive_root/main/tools/rna_tools" \
        "$archive_root/main/source/bin/rna_denovo.static.linuxgccrelease" \
        "$archive_root/main/source/bin/extract_pdbs.static.linuxgccrelease" \
        "$archive_root/$build_path/rna_denovo.static.linuxgccrelease" \
        "$archive_root/$build_path/extract_pdbs.static.linuxgccrelease" \
        "$archive_root/main/LICENSE.md"
    runtime=$tmp_dir/extracted/$archive_root
    mv "$runtime/main/LICENSE.md" "$tmp_dir/LICENSE.md"
    check_sha "$tmp_dir/LICENSE.md" "$license_sha"
fi

# Bytecode created by Python imports is not part of the released runtime.
(cd "$runtime" && find . -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$tmp_dir/runtime-SHA256SUMS"
check_sha "$tmp_dir/runtime-SHA256SUMS" "$runtime_sha"
check_sha "$runtime/main/source/bin/rna_denovo.static.linuxgccrelease" "$rna_sha"
check_sha "$runtime/main/source/bin/extract_pdbs.static.linuxgccrelease" "$extract_sha"
if [ ! -e "$runtime/database" ]; then
    ln -s main/database "$runtime/database"
fi
if [ ! -e "$runtime/main/source/bin/extract_pdbs.linuxgccrelease" ]; then
    ln -s extract_pdbs.static.linuxgccrelease "$runtime/main/source/bin/extract_pdbs.linuxgccrelease"
fi
(cd "$runtime" && find . -type l -printf '%p -> %l\n' | LC_ALL=C sort) > "$tmp_dir/symlinks.txt"
check_sha "$tmp_dir/symlinks.txt" "$symlinks_sha"
# This executable prints its version, then exits without an input sequence.
version_status=0
(cd "$tmp_dir" && "$runtime/main/source/bin/rna_denovo.static.linuxgccrelease" -version) > "$tmp_dir/version.txt" 2>&1 || version_status=$?
grep -Fq '2024.09+release.06b3cf8' "$tmp_dir/version.txt"
grep -Fq '06b3cf8ad0940d628690d0ed6fa2009d72ad2b44' "$tmp_dir/version.txt"
if [ "$runtime" != "$out_dir" ]; then
    mkdir -p "$(dirname "$out_dir")"
    mv "$runtime" "$out_dir"
fi
cp "$tmp_dir/image-manifest.json" "$out_dir.image-manifest.json"
cp "$tmp_dir/runtime-SHA256SUMS" "$out_dir.SHA256SUMS"
cp "$tmp_dir/version.txt" "$out_dir.version.txt"
cp "$tmp_dir/symlinks.txt" "$out_dir.symlinks.txt"
if [ -f "$tmp_dir/LICENSE.md" ]; then
    cp "$tmp_dir/LICENSE.md" "$out_dir.LICENSE.md"
fi
python3 - "$out_dir.provenance.json" "$image_sha" "$layer_sha" "$runtime_sha" "$version_status" "$symlinks_sha" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'image': 'tanwenchong/ensirna@sha256:' + sys.argv[2],
    'application_layer_sha256': sys.argv[3], 'runtime_manifest_sha256': sys.argv[4],
    'version': '2024.09+release.06b3cf8', 'release': 371,
    'git_commit': '06b3cf8ad0940d628690d0ed6fa2009d72ad2b44',
    'version_probe_exit_code': int(sys.argv[5]), 'runtime_symlinks_sha256': sys.argv[6]}, indent=2) + '\n')
PY
printf 'Verified ENsiRNA Rosetta release-371 / 2024.09+release.06b3cf8 at %s\n' "$out_dir"
