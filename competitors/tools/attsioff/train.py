#!/usr/bin/env python
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from scipy import stats


def load_modules(src_root):
    sys.path.insert(0, src_root)
    from model import RNAFM_SIPRED_2
    from load_data import Generate_dataset, create_pssm
    return RNAFM_SIPRED_2, Generate_dataset, create_pssm


def _col_or_zeros(df, col):
    if col in df.columns:
        return df[col]
    return np.zeros(len(df))


def build_dataset(df, create_pssm):
    seq = np.array(df['Antisense'])
    data = {
        'seq': seq,
        'mrna': np.array(df['mrna']),
        's-biopredsi': np.array(_col_or_zeros(df, 's-Biopredsi')),
        'dsir': np.array(_col_or_zeros(df, 'DSIR')) / 100.0,
        'i-score': np.array(_col_or_zeros(df, 'i-score')) / 100.0,
        'inhibition': np.array(df['inhibition']),
        'RNAFM_ind': np.array(df['RNAFM_ind']),
    }
    pssm = create_pssm(data['seq'])
    return data, pssm


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        for k in batch:
            if hasattr(batch[k], 'to'):
                batch[k] = batch[k].to(device)
        pred = model(batch)
        label = batch['inhibit']
        loss = criterion(label, pred)
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
    labels = []
    preds = []
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)
            pred = model(batch)
            label = batch['inhibit']
            loss = criterion(label, pred)
            total_loss += loss.item() * label.shape[0]
            count += label.shape[0]
            labels.extend(label.detach().cpu().numpy().tolist())
            preds.extend(pred.detach().cpu().numpy().tolist())
    pcc = None
    try:
        if len(labels) > 1:
            pcc = stats.pearsonr(preds, labels)[0]
    except Exception:
        pcc = None
    return total_loss / max(count, 1), pcc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--data-dir")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--cuda", default="0")
    p.add_argument("--early-stopping", type=int, default=20)
    p.add_argument("--src-root", default="attsioff_src")
    args = p.parse_args()

    data_dir = args.data_dir or os.path.dirname(os.path.abspath(args.train_csv))
    os.chdir(data_dir)
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), args.src_root))
    RNAFM_SIPRED_2, Generate_dataset, create_pssm = load_modules(src_root)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    train_data, pssm_train = build_dataset(train_df, create_pssm)
    val_data, pssm_val = build_dataset(val_df, create_pssm)

    trainset = Generate_dataset(args, train_data, pssm_train)
    valset = Generate_dataset(args, val_data, pssm_val)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=valset.__len__(), shuffle=False, drop_last=False)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model = RNAFM_SIPRED_2(dp=0.1, device=device).to(torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.MSELoss(reduction='mean')

    os.makedirs(args.model_dir, exist_ok=True)
    best_loss = None
    best_pcc = None
    best_epoch = -1
    best_path = os.path.join(args.model_dir, "model.pt")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_pcc = eval_epoch(model, val_loader, criterion, device)
        improved = False
        if val_pcc is not None:
            if best_pcc is None or val_pcc > best_pcc:
                improved = True
        else:
            if best_loss is None or val_loss < best_loss:
                improved = True
        if improved:
            best_loss = val_loss
            best_pcc = val_pcc
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_pcc={val_pcc}")
        if epoch - best_epoch > args.early_stopping:
            break

    meta = {
        "train_csv": os.path.abspath(args.train_csv),
        "val_csv": os.path.abspath(args.val_csv) if args.val_csv else None,
        "model_path": os.path.abspath(best_path),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "early_stopping": args.early_stopping,
    }
    with open(os.path.join(args.model_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
