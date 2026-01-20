#!/usr/bin/env python
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from stellargraph.mapper import HinSAGENodeGenerator
from stellargraph.layer import HinSAGE

from common import build_graph, load_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--preprocess-dir", required=True)
    p.add_argument("--rna-ago2-dir", required=True)
    p.add_argument("--params-json", default="params.json")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--src-root", default="sirnadiscovery_src")
    args = p.parse_args()

    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), args.src_root))
    params = load_params(args.params_json)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    data_all = pd.concat([train_df, val_df], axis=0)
    graph = build_graph(data_all, args.preprocess_dir, args.rna_ago2_dir, params, src_root)

    generator = HinSAGENodeGenerator(graph, params["batch_size"], params["hop_samples"], head_node_type="interaction")
    hinsage = HinSAGE(layer_sizes=params["hinsage_layer_sizes"], generator=generator, bias=True, dropout=params["dropout"])
    x_inp, x_out = hinsage.in_out_tensors()

    prediction = tf.keras.layers.Dense(units=1)(x_out)
    model = tf.keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]), loss=params["loss"])

    train_interaction = pd.DataFrame(train_df['efficacy'].values, index=train_df['siRNA'] + "_" + train_df['mRNA'])
    val_interaction = pd.DataFrame(val_df['efficacy'].values, index=val_df['siRNA'] + "_" + val_df['mRNA'])

    train_gen = generator.flow(train_interaction.index, train_interaction, shuffle=True)
    val_gen = generator.flow(val_interaction.index, val_interaction)

    model.fit(train_gen, epochs=params["epochs"], validation_data=val_gen, verbose=2, shuffle=False)

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.keras")
    model.save(model_path)

    meta = {
        "train_csv": os.path.abspath(args.train_csv),
        "val_csv": os.path.abspath(args.val_csv),
        "test_csv": None,
        "preprocess_dir": os.path.abspath(args.preprocess_dir),
        "rna_ago2_dir": os.path.abspath(args.rna_ago2_dir),
        "model_path": os.path.abspath(model_path),
        "params": params,
    }
    with open(os.path.join(args.model_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
