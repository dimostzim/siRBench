#!/usr/bin/env python
import argparse
import os
import sys

from runner import TOOL_CHOICES, tool_dir, repo_root, to_container_path, run_docker

PATH_FLAGS = {
    "--train-csv",
    "--val-csv",
    "--test-csv",
    "--data-dir",
    "--model-dir",
    "--output-dir",
    "--work-dir",
    "--fasta-dir",
    "--rnafm-dir",
    "--bert-dir",
    "--train-set",
    "--valid-set",
    "--processed-dir",
    "--preprocess-dir",
    "--rna-ago2-dir",
    "--params-json",
}


def rewrite_args(argv, host_root):
    out = []
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tool", required=True, choices=TOOL_CHOICES)
    p.add_argument("--docker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--gpus", default="all", help="GPU selector for Docker, e.g. all or 0")
    args, unknown = p.parse_known_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tdir = tool_dir(base_dir, args.tool)
    script_path = os.path.join(tdir, "train.py")

    if not os.path.exists(script_path):
        print(f"train.py not found for tool: {args.tool}", file=sys.stderr)
        sys.exit(1)

    host_root = repo_root(base_dir)
    forwarded = rewrite_args(unknown, host_root)
    run_docker(args.tool, "train.py", forwarded, host_root, gpus=args.gpus)


if __name__ == "__main__":
    main()
