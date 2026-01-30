#!/usr/bin/env python
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


def load_modules(src_root):
    sys.path.insert(0, os.path.join(src_root, "scripts"))
    from loader import data_process_loader
    from model import Oligo
    return data_process_loader, Oligo


def seed_everything(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        siRNA, mRNA, siRNA_FM, mRNA_FM, label, _, td = batch
        siRNA = siRNA.to(device)
        mRNA = mRNA.to(device)
        siRNA_FM = siRNA_FM.to(device)
        mRNA_FM = mRNA_FM.to(device)
        label = label.to(device).float()
        td = td.to(device)

        pred, _, _ = model(siRNA, mRNA, siRNA_FM, mRNA_FM, td)
        loss = criterion(pred[:, 1], label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * label.shape[0]
        count += label.shape[0]
    return total_loss / max(count, 1)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            siRNA, mRNA, siRNA_FM, mRNA_FM, label, y, td = batch
            siRNA = siRNA.to(device)
            mRNA = mRNA.to(device)
            siRNA_FM = siRNA_FM.to(device)
            mRNA_FM = mRNA_FM.to(device)
            label = label.to(device).float()
            td = td.to(device)

            pred, _, _ = model(siRNA, mRNA, siRNA_FM, mRNA_FM, td)
            loss = criterion(pred[:, 1], label)
            all_labels.extend(y.detach().cpu().numpy().tolist())
            all_probs.extend(pred[:, 1].detach().cpu().numpy().tolist())
            total_loss += loss.item() * label.shape[0]
            count += label.shape[0]
    avg_loss = total_loss / max(count, 1)
    auc = None
    try:
        auc = roc_auc_score(np.array(all_labels), np.array(all_probs))
    except Exception:
        auc = None
    return avg_loss, auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--data-dir")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--train-name")
    p.add_argument("--val-name")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.999)
    p.add_argument("--early-stopping", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--cuda", default="0")
    p.add_argument("--oligoformer-src", default="oligoformer_src")
    args = p.parse_args()

    seed_everything(args.seed, args.deterministic)

    data_dir = args.data_dir or os.path.dirname(os.path.abspath(args.train_csv))
    train_name = args.train_name or os.path.splitext(os.path.basename(args.train_csv))[0]
    val_name = args.val_name
    if args.val_csv and not val_name:
        val_name = os.path.splitext(os.path.basename(args.val_csv))[0]

    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), args.oligoformer_src))
    data_process_loader, Oligo = load_modules(src_root)

    train_df = pd.read_csv(args.train_csv, dtype=str)
    val_df = pd.read_csv(args.val_csv, dtype=str)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": 0,
        "drop_last": False,
    }

    train_loader = DataLoader(
        data_process_loader(train_df.index.values, train_df.label.values, train_df.y.values, train_df, train_name, data_dir),
        **params,
    )
    val_loader = DataLoader(
        data_process_loader(val_df.index.values, val_df.label.values, val_df.y.values, val_df, val_name or train_name, data_dir),
        **params,
    )

    model = Oligo().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.weight_decay)
    criterion = nn.MSELoss()

    os.makedirs(args.model_dir, exist_ok=True)
    best_loss = None
    best_auc = -1.0
    best_epoch = -1
    best_path = os.path.join(args.model_dir, "model.pt")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        improved = False
        if val_auc is not None:
            if best_loss is None or (val_loss < best_loss and val_auc > best_auc):
                improved = True
        else:
            if best_loss is None or val_loss < best_loss:
                improved = True
        if improved:
            best_loss = val_loss
            best_auc = val_auc if val_auc is not None else best_auc
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_auc={val_auc}")
        if epoch - best_epoch > args.early_stopping:
            break

    meta = {
        "train_csv": os.path.abspath(args.train_csv),
        "val_csv": os.path.abspath(args.val_csv) if args.val_csv else None,
        "train_name": train_name,
        "val_name": val_name or train_name,
        "data_dir": os.path.abspath(data_dir),
        "model_path": os.path.abspath(best_path),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "early_stopping": args.early_stopping,
        "seed": args.seed,
    }
    with open(os.path.join(args.model_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
