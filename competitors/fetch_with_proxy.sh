#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="${ROOT_DIR}/tools"

proxy_args=()
if [ -n "${http_proxy:-}" ]; then
  proxy_args+=(-c "http.proxy=${http_proxy}")
fi
if [ -n "${https_proxy:-}" ]; then
  proxy_args+=(-c "https.proxy=${https_proxy}")
fi
if [ -n "${HTTP_PROXY:-}" ]; then
  proxy_args+=(-c "http.proxy=${HTTP_PROXY}")
fi
if [ -n "${HTTPS_PROXY:-}" ]; then
  proxy_args+=(-c "https.proxy=${HTTPS_PROXY}")
fi

clone_repo() {
  local repo="$1"
  local dest="$2"
  if [ -d "${dest}" ]; then
    echo "Exists: ${dest}"
    return
  fi
  echo "Cloning ${repo} -> ${dest}"
  git "${proxy_args[@]}" clone "${repo}" "${dest}"
}

clone_repo "https://github.com/lulab/OligoFormer.git" "${TOOLS_DIR}/oligoformer/oligoformer_src"
clone_repo "https://github.com/ml4bio/RNA-FM.git" "${TOOLS_DIR}/oligoformer/oligoformer_src/RNA-FM"

clone_repo "https://github.com/2333liubin/AttSiOff.git" "${TOOLS_DIR}/attsioff/attsioff_src"
clone_repo "https://github.com/ml4bio/RNA-FM.git" "${TOOLS_DIR}/attsioff/attsioff_src/RNA-FM"

clone_repo "https://github.com/ChengkuiZhao/siRNABERT.git" "${TOOLS_DIR}/sirnabert/sirnabert_src"
clone_repo "https://github.com/tanwenchong/ENsiRNA.git" "${TOOLS_DIR}/ensirna/ensirna_src"
clone_repo "https://github.com/BCB4PM/GNN4siRNA.git" "${TOOLS_DIR}/gnn4sirna/gnn4sirna_src"
clone_repo "https://github.com/BertramLoong/siRNADiscovery.git" "${TOOLS_DIR}/sirnadiscovery/sirnadiscovery_src"

echo "Done."
