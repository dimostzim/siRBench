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
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--final-lr", type=float, default=None)
    p.add_argument("--max-epoch", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--save-topk", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--original-params", action="store_true", help="Use upstream default hyperparameters.")
    p.add_argument("--src-root", default="ensirna_src")
    args, unknown = p.parse_known_args()

    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), args.src_root, "ENsiRNA"))
    train_script = os.path.join(src_root, "train.py")

    os.makedirs(args.model_dir, exist_ok=True)

    if args.original_params:
        args.lr = 1e-3
        args.final_lr = 1e-4
        args.max_epoch = 10
        args.patience = 1000
        args.save_topk = 10
        if args.seed is None:
            args.seed = 12

    if "--embed_dim" not in unknown:
        unknown += ["--embed_dim", "128"]
    if "--num_workers" not in unknown:
        unknown += ["--num_workers", "0"]
    if args.seed is not None:
        os.environ["ENSIRNA_SEED"] = str(args.seed)
    if args.lr is not None and "--lr" not in unknown:
        unknown += ["--lr", str(args.lr)]
    if args.final_lr is not None and "--final_lr" not in unknown:
        unknown += ["--final_lr", str(args.final_lr)]
    if args.max_epoch is not None and "--max_epoch" not in unknown:
        unknown += ["--max_epoch", str(args.max_epoch)]
    if args.patience is not None and "--patience" not in unknown:
        unknown += ["--patience", str(args.patience)]
    if args.save_topk is not None and "--save_topk" not in unknown:
        unknown += ["--save_topk", str(args.save_topk)]

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
