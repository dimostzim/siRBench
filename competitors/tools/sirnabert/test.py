#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from metrics import format_metrics, regression_metrics, save_metrics


def seq2kmer(seq, k=6):
    return " ".join([seq[i:i+k] for i in range(len(seq) + 1 - k)])


class SiRNADataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.labels = df['efficacy'].astype(float).tolist()
        self.texts = [seq2kmer(s.replace('U', 'T'), 6) for s in df['siRNA']]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return enc, label


class siRNABertRegressor(torch.nn.Module):
    def __init__(self, bert_dir, dropout=0.1):
        super().__init__()
        config = BertConfig.from_pretrained(bert_dir, output_attentions=True)
        self.bert = BertModel.from_pretrained(bert_dir, config=config)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear1 = torch.nn.Linear(768, 128)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128, 128)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(128, 64)
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(64, 1)
        self.sigmoid4 = torch.nn.Sigmoid()

    def forward(self, input_id, mask):
        output = self.bert(input_ids=input_id, attention_mask=mask)
        label_output = output[0][:, 0, :]
        dropout_output = self.dropout(label_output)
        linear_output1 = self.linear1(dropout_output)
        reluoutput1 = self.relu1(linear_output1)
        linear_output2 = self.linear2(reluoutput1)
        reluoutput2 = self.relu2(linear_output2)
        linear_output3 = self.linear3(reluoutput2)
        reluoutput3 = self.relu3(linear_output3)
        linear_output4 = self.linear4(reluoutput3)
        final_layer = self.sigmoid4(linear_output4)
        return final_layer.squeeze(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-csv", required=True)
    p.add_argument("--bert-dir", default=os.environ.get("BERT_DIR", "/opt/dnabert/6mer"))
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-csv", default="predictions.csv")
    p.add_argument("--metrics-json", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=16)
    p.add_argument("--cuda", default="0")
    args = p.parse_args()

    df = pd.read_csv(args.test_csv)
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    ds = SiRNADataset(df, tokenizer, args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model = siRNABertRegressor(args.bert_dir).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for enc, _label in loader:
            input_id = enc['input_ids'].squeeze(1).to(device)
            mask = enc['attention_mask'].to(device)
            pred = model(input_id, mask)
            preds.extend(pred.detach().cpu().numpy().tolist())

    out_df = pd.DataFrame({
        "id": df.get("id", pd.Series([f"row_{i}" for i in range(len(df))])),
        "label": df["efficacy"].astype(float),
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
