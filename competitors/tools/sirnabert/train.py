#!/usr/bin/env python
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig, BertTokenizer


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


class siRNABertRegressor(nn.Module):
    def __init__(self, bert_dir, dropout=0.1):
        super().__init__()
        config = BertConfig.from_pretrained(bert_dir, output_attentions=True)
        self.bert = BertModel.from_pretrained(bert_dir, config=config)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(64, 1)
        self.sigmoid4 = nn.Sigmoid()

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
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--bert-dir", default=os.environ.get("BERT_DIR", "/opt/dnabert/6mer"))
    p.add_argument("--model-dir", default="models")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-len", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--cuda", default="0")
    args = p.parse_args()

    seed_everything(args.seed, args.deterministic)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    train_ds = SiRNADataset(train_df, tokenizer, args.max_len)
    val_ds = SiRNADataset(val_df, tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model = siRNABertRegressor(args.bert_dir).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.MSELoss()

    os.makedirs(args.model_dir, exist_ok=True)
    best_loss = None
    best_path = os.path.join(args.model_dir, "model.pt")

    for epoch in range(args.epochs):
        model.train()
        for enc, label in train_loader:
            input_id = enc['input_ids'].squeeze(1).to(device)
            mask = enc['attention_mask'].to(device)
            label = label.to(device)
            pred = model(input_id, mask)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for enc, label in val_loader:
                input_id = enc['input_ids'].squeeze(1).to(device)
                mask = enc['attention_mask'].to(device)
                label = label.to(device)
                pred = model(input_id, mask)
                loss = criterion(pred, label)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)
        print(f"epoch={epoch} val_loss={val_loss:.6f}")

    meta = {
        "train_csv": os.path.abspath(args.train_csv),
        "val_csv": os.path.abspath(args.val_csv) if args.val_csv else None,
        "bert_dir": args.bert_dir,
        "model_path": os.path.abspath(best_path),
    }
    with open(os.path.join(args.model_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
