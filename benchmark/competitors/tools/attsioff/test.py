#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from metrics import format_metrics, regression_metrics, save_metrics


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
    data = {
        'seq': np.array(df['Antisense']),
        'mrna': np.array(df['mrna']),
        's-biopredsi': np.array(_col_or_zeros(df, 's-Biopredsi')),
        'dsir': np.array(_col_or_zeros(df, 'DSIR')) / 100.0,
        'i-score': np.array(_col_or_zeros(df, 'i-score')) / 100.0,
        'inhibition': np.array(df['inhibition']),
        'RNAFM_ind': np.array(df['RNAFM_ind']),
    }
    pssm = create_pssm(data['seq'])
    return data, pssm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-csv", required=True)
    p.add_argument("--data-dir")
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-csv", default="predictions.csv")
    p.add_argument("--metrics-json", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cuda", default="0")
    p.add_argument("--src-root", default="attsioff_src")
    args = p.parse_args()

    data_dir = args.data_dir or os.path.dirname(os.path.abspath(args.test_csv))
    os.chdir(data_dir)
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), args.src_root))
    RNAFM_SIPRED_2, Generate_dataset, create_pssm = load_modules(src_root)

    df = pd.read_csv(args.test_csv)
    data, pssm = build_dataset(df, create_pssm)
    testset = Generate_dataset(args, data, pssm)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model = RNAFM_SIPRED_2(dp=0.1, device=device).to(torch.float32).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in test_loader:
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)
                    if batch[k].dtype == torch.float64:
                        batch[k] = batch[k].to(torch.float32)
            pred = model(batch)
            preds.extend(pred.detach().cpu().numpy().tolist())

    out_df = pd.DataFrame({
        "id": df.get("id", pd.Series([f"row_{i}" for i in range(len(df))])),
        "label": df["inhibition"].astype(float),
        "pred_label": preds,
    })
    out_df.to_csv(args.output_csv, index=False)
    metrics = regression_metrics(out_df["label"].to_numpy(), out_df["pred_label"].to_numpy())
    if args.metrics_json:
        save_metrics(metrics, args.metrics_json)
    print(format_metrics(metrics))
    print(args.output_csv)


if __name__ == "__main__":
    main()
