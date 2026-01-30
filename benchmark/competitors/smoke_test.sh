#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

TOOLS=(oligoformer gnn4sirna sirnadiscovery attsioff sirnabert ensirna)

for tool in "${TOOLS[@]}"; do
  echo "=== ${tool} ==="
  if [ -z "${SKIP_BUILD:-}" ]; then
    "${ROOT_DIR}/setup.sh" --tool "$tool"
  fi
  docker run --rm -v "${REPO_ROOT}:/work" -w "/work/competitors/tools/${tool}" "${tool}:latest" python3 prepare.py --help >/dev/null
  docker run --rm -v "${REPO_ROOT}:/work" -w "/work/competitors/tools/${tool}" "${tool}:latest" python3 train.py --help >/dev/null
  docker run --rm -v "${REPO_ROOT}:/work" -w "/work/competitors/tools/${tool}" "${tool}:latest" python3 test.py --help >/dev/null
  echo "ok"
done
