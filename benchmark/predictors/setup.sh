#!/bin/bash
set -e

TOOLS=()
RUN_ALL=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --tool)
            shift
            if [ $# -eq 0 ] || [[ "$1" == --* ]]; then
                echo "Missing value for --tool"
                exit 1
            fi
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                TOOLS+=("$1")
                shift
            done
            RUN_ALL=0
            ;;
        --docker)
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--tool <name>...]"
            echo "If no flags provided, sets up all tools."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ "$RUN_ALL" = "1" ]; then
    TOOLS=(oligoformer sirnadiscovery sirnabert attsioff gnn4sirna ensirna)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/tools"
for tool in "${TOOLS[@]}"; do
    TOOL_DIR="${BASE_DIR}/${tool}"
    if [ ! -d "${TOOL_DIR}" ]; then
        echo "Unknown tool: ${tool}"
        exit 1
    fi
    (cd "${TOOL_DIR}" && ./setup.sh --docker)
done
