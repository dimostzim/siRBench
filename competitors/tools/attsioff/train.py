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
    import load_data as ld
    from model import RNAFM_SIPRED_2

    def pad_rnafm(arr, target_len):
        pad_dtype = arr.dtype
        cur_len = arr.shape[0]
        if cur_len < target_len:
            pad = target_len - cur_len
            pad_front = pad // 2
            pad_back = pad - pad_front
            if pad_front:
                arr = np.concatenate([np.ones((pad_front, 640), dtype=pad_dtype) * 0.05, arr], axis=0)
            if pad_back:
                arr = np.concatenate([arr, np.ones((pad_back, 640), dtype=pad_dtype) * 0.05], axis=0)
        elif cur_len > target_len:
            extra = cur_len - target_len
            start = extra // 2
            arr = arr[start:start + target_len]
        return arr

    def process_mrna_RNAFM(ori_mrna, rnafm_mrna):
        pad_dtype = rnafm_mrna.dtype
        front_dot_num = ori_mrna[:20].count('.')
        back_dot_num = ori_mrna[-20:].count('.')
        if front_dot_num:
            rnafm_mrna = np.concatenate([np.ones((front_dot_num, 640), dtype=pad_dtype) * 0.05, rnafm_mrna], axis=0)
        if back_dot_num:
            rnafm_mrna = np.concatenate([rnafm_mrna, np.ones((back_dot_num, 640), dtype=pad_dtype) * 0.05], axis=0)
        return pad_rnafm(rnafm_mrna, 59)

    def process_sirna_RNAFM(rnafm_sirna):
        return pad_rnafm(rnafm_sirna, 21)

    ld.process_mrna_RNAFM = process_mrna_RNAFM
    orig_load_sirna = ld.load_sirna

    def load_sirna(args, dataset, pssm):
        result = orig_load_sirna(args, dataset, pssm)
        for item in result:
            item['rnafm_encode'] = process_sirna_RNAFM(item['rnafm_encode']).astype(np.float32)
            item['rnafm_encode_mrna'] = item['rnafm_encode_mrna'].astype(np.float32)
        return result

    ld.load_sirna = load_sirna
    return RNAFM_SIPRED_2, ld.Generate_dataset, ld.create_pssm


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
                if batch[k].dtype == torch.float64:
                    batch[k] = batch[k].to(torch.float32)
        pred = model(batch)
        label = batch['inhibit'].to(torch.float32)
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
                    if batch[k].dtype == torch.float64:
                        batch[k] = batch[k].to(torch.float32)
            pred = model(batch)
            label = batch['inhibit'].to(torch.float32)
            loss = criterion(label, pred)
            total_loss += loss.item() * label.shape[0]
            count += label.shape[0]
            labels.extend(label.detach().cpu().numpy().tolist())
            preds.extend(pred.detach().cpu().numpy().tolist())
    pcc = None
    spcc = None
    try:
        if len(labels) > 1:
            pcc = stats.pearsonr(preds, labels)[0]
            spcc = stats.spearmanr(preds, labels)[0]
    except Exception:
        pass
    return total_loss / max(count, 1), pcc, spcc


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
    best_spcc = None
    best_epoch = -1
    best_path = os.path.join(args.model_dir, "model.pt")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_pcc, val_spcc = eval_epoch(model, val_loader, criterion, device)
        # Model selection / early stopping: optimize Spearman correlation (higher is better).
        # This matches the original AttSiOff implementation.
        improved = best_spcc is None or (val_spcc is not None and val_spcc > best_spcc)
        if improved:
            best_spcc = val_spcc
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_pcc={val_pcc} val_spcc={val_spcc}")
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
