#!/usr/bin/env python
import argparse
import os
import sys

from runner import TOOL_CHOICES, tool_dir, repo_root, to_container_path, run_docker

PATH_FLAGS = {
    "--input-csv",
    "--output-dir",
    "--output-jsonl",
    "--data-dir",
    "--work-dir",
    "--fasta-dir",
    "--rnafm-dir",
    "--rnafm-root",
    "--pdb-dir",
    "--rosetta-dir",
    "--model-dir",
    "--preprocess-dir",
    "--rna-ago2-dir",
    "--processed-dir",
}


def rewrite_args(argv, host_root):
    out = []
    it = iter(range(len(argv)))
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in PATH_FLAGS and i + 1 < len(argv):
            out.append(arg)
            out.append(to_container_path(argv[i + 1], host_root))
            i += 2
            continue
        out.append(arg)
        i += 1
    return out


def get_arg_value(argv, flag):
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tool", required=True, choices=TOOL_CHOICES)
    p.add_argument("--docker", action="store_true", help=argparse.SUPPRESS)
    args, unknown = p.parse_known_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tdir = tool_dir(base_dir, args.tool)
    script_path = os.path.join(tdir, "prepare.py")

    if not os.path.exists(script_path):
        print(f"prepare.py not found for tool: {args.tool}", file=sys.stderr)
        sys.exit(1)

    host_root = repo_root(base_dir)
    forwarded = rewrite_args(unknown, host_root)

    input_csv = get_arg_value(unknown, "--input-csv") or "data"
    dataset_name = get_arg_value(unknown, "--dataset-name") or os.path.basename(input_csv)
    status_msg = f"[{args.tool}] prepare {dataset_name}"

    run_docker(args.tool, "prepare.py", forwarded, host_root, status_msg=status_msg)


if __name__ == "__main__":
    main()
