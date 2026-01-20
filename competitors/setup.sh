#!/bin/bash
set -e

TOOL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --tool)
            TOOL="$2"
            shift 2
            ;;
        --docker)
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$TOOL" ]; then
    echo "Error: --tool is required"
    exit 1
fi

cd "$(dirname "$0")/tools/$TOOL"
./setup.sh --docker
