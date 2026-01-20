#!/usr/bin/env python
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import stellargraph as sg
from stellargraph.mapper import HinSAGENodeGenerator
from stellargraph.layer import HinSAGE


def load_processed(processed_dir):
    sirna_kmers = pd.read_csv(os.path.join(processed_dir, "sirna_kmers.txt"), header=None).set_index(0)
    mrna_path = os.path.join(processed_dir, "target_kmers.txt")
    if not os.path.exists(mrna_path):
        mrna_path = os.path.join(processed_dir, "mRNA_kmers.txt")
    mrna_kmers = pd.read_csv(mrna_path, header=None).set_index(0)

    thermo = pd.read_csv(os.path.join(processed_dir, "sirna_target_thermo.csv"), header=None)
    thermo.rename(columns={0: "source", 1: "target"}, inplace=True)
    interaction = thermo.drop(["source", "target"], axis=1)
    interaction["index"] = thermo["source"].astype(str) + "_" + thermo["target"].astype(str)
    interaction = interaction.set_index("index")

    edges1 = pd.DataFrame({"source": interaction.index, "target": thermo["source"]})
    edges2 = pd.DataFrame({"source": interaction.index, "target": thermo["target"]})
    edges = pd.concat([edges1, edges2], ignore_index=True)

    graph = sg.StellarGraph({"siRNA": sirna_kmers, "mRNA": mrna_kmers, "interaction": interaction},
                            edges=edges, source_column="source", target_column="target")
    return graph


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--test-csv")
    p.add_argument("--processed-dir", required=True)
    p.add_argument("--model-dir", default="models")
    p.add_argument("--batch-size", type=int, default=60)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", default="mse")
    args = p.parse_args()

    graph = load_processed(args.processed_dir)

    train_df = pd.read_csv(args.train_csv)
    if args.test_csv:
        raise ValueError("--test-csv is not allowed during training; use --val-csv and run test separately.")
    val_df = pd.read_csv(args.val_csv)
    val_source = "val_csv"

    train_interaction = pd.DataFrame(train_df["efficacy"].values,
                                     index=train_df["siRNA"] + "_" + train_df["mRNA"])
    val_interaction = pd.DataFrame(val_df["efficacy"].values,
                                   index=val_df["siRNA"] + "_" + val_df["mRNA"])

    generator = HinSAGENodeGenerator(graph, args.batch_size, [8, 4], head_node_type="interaction")
    train_gen = generator.flow(train_interaction.index, train_interaction, shuffle=True)
    val_gen = generator.flow(val_interaction.index, val_interaction)

    hinsage = HinSAGE(layer_sizes=[32, 16], generator=generator, bias=True, dropout=0.15)
    x_inp, x_out = hinsage.in_out_tensors()
    prediction = tf.keras.layers.Dense(units=1)(x_out)
    model = tf.keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss=args.loss)

    model.fit(train_gen, epochs=args.epochs, validation_data=val_gen, verbose=2, shuffle=False)

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.keras")
    model.save(model_path)

    meta = {
        "train_csv": os.path.abspath(args.train_csv),
        "val_csv": os.path.abspath(args.val_csv) if args.val_csv else None,
        "test_csv": None,
        "val_source": val_source,
        "processed_dir": os.path.abspath(args.processed_dir),
        "model_path": os.path.abspath(model_path),
    }
    with open(os.path.join(args.model_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
