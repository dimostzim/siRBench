#!/usr/bin/env python
import argparse
import os
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-set", required=True)
    p.add_argument("--valid-set", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gpus", nargs='+', default=["0"])
    p.add_argument("--model-type", default="RNAmaskModel")
    p.add_argument("--src-root", default="ensirna_src")
    args, unknown = p.parse_known_args()

    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), args.src_root, "ENsiRNA"))
    train_script = os.path.join(src_root, "train.py")

    os.makedirs(args.model_dir, exist_ok=True)

    cmd = [
        sys.executable, train_script,
        "--train_set", args.train_set,
        "--valid_set", args.valid_set,
        "--save_dir", args.model_dir,
        "--batch_size", str(args.batch_size),
        "--model_type", args.model_type,
        "--gpus",
    ] + args.gpus + [
    ] + unknown

    subprocess.check_call(cmd, cwd=src_root)


if __name__ == "__main__":
    main()
