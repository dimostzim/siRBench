#!/usr/bin/env python
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
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
    p.add_argument("--src-root", default="sirnadiscovery_src/siRNA_split")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--loss", default=None)
    p.add_argument("--early-stopping", type=int, default=0, help="Patience for early stopping; 0 disables.")
    p.add_argument("--early-stop-metric", default="val_loss", choices=["val_loss", "val_r2_metric"])
    p.add_argument("--early-stop-mode", default="auto", choices=["auto", "min", "max"])
    p.add_argument("--original-params", action="store_true", help="Use upstream params.json without overrides.")
    p.add_argument("--allow-missing-preprocess", action="store_true", help="Fill missing preprocess rows with zeros.")
    p.add_argument("--allow-missing-ago2", action="store_true", help="Fill missing RNA_AGO2 rows with zeros.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    if args.deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    import tensorflow as tf

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    if args.deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), args.src_root))
    params = load_params(args.params_json)
    if not args.original_params:
        if args.epochs is not None:
            params["epochs"] = args.epochs
        if args.batch_size is not None:
            params["batch_size"] = args.batch_size
        if args.lr is not None:
            params["lr"] = args.lr
        if args.loss is not None:
            params["loss"] = args.loss

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    data_all = pd.concat([train_df, val_df], axis=0)
    graph = build_graph(
        data_all,
        args.preprocess_dir,
        args.rna_ago2_dir,
        params,
        src_root,
        allow_missing=args.allow_missing_preprocess,
        allow_missing_ago2=args.allow_missing_ago2,
    )

    generator = HinSAGENodeGenerator(graph, params["batch_size"], params["hop_samples"], head_node_type="interaction")
    hinsage = HinSAGE(layer_sizes=params["hinsage_layer_sizes"], generator=generator, bias=True, dropout=params["dropout"])
    x_inp, x_out = hinsage.in_out_tensors()

    prediction = tf.keras.layers.Dense(units=1)(x_out)
    model = tf.keras.Model(inputs=x_inp, outputs=prediction)
    def r2_metric(y_true, y_pred):
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        return tf.where(tf.equal(ss_tot, 0.0), 0.0, 1.0 - ss_res / ss_tot)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]),
        loss=params["loss"],
        metrics=[r2_metric],
    )

    train_interaction = pd.DataFrame(train_df['efficiency'].values, index=train_df['siRNA'] + "_" + train_df['mRNA'])
    val_interaction = pd.DataFrame(val_df['efficiency'].values, index=val_df['siRNA'] + "_" + val_df['mRNA'])

    train_gen = generator.flow(train_interaction.index, train_interaction, shuffle=True)
    val_gen = generator.flow(val_interaction.index, val_interaction)

    callbacks = []
    if args.early_stopping and args.early_stopping > 0:
        mode = args.early_stop_mode
        if mode == "auto":
            mode = "min" if "loss" in args.early_stop_metric else "max"
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=args.early_stop_metric,
                patience=args.early_stopping,
                mode=mode,
                restore_best_weights=True,
            )
        )

    model.fit(train_gen, epochs=params["epochs"], validation_data=val_gen, verbose=2, shuffle=False, callbacks=callbacks)

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
