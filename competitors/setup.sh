#!/bin/bash
set -e

TOOLS=()

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
            ;;
        --docker)
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --tool <name>..."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ ${#TOOLS[@]} -eq 0 ]; then
    echo "Error: --tool is required"
    exit 1
fi

BASE_DIR="$(dirname "$0")/tools"
for tool in "${TOOLS[@]}"; do
    TOOL_DIR="${BASE_DIR}/${tool}"
    if [ ! -d "${TOOL_DIR}" ]; then
        echo "Unknown tool: ${tool}"
        exit 1
    fi
    (cd "${TOOL_DIR}" && ./setup.sh --docker)
done
