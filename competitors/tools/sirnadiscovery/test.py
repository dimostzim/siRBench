#!/usr/bin/env python
import argparse
import os
import sys

import pandas as pd
import tensorflow as tf
from stellargraph.mapper import HinSAGENodeGenerator

from common import build_graph, load_params

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
from metrics import format_metrics, regression_metrics, save_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-csv", required=True)
    p.add_argument("--preprocess-dir", required=True)
    p.add_argument("--rna-ago2-dir", required=True)
    p.add_argument("--params-json", default="params.json")
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-csv", default="predictions.csv")
    p.add_argument("--metrics-json", default=None)
    p.add_argument("--src-root", default="sirnadiscovery_src")
    args = p.parse_args()

    params = load_params(args.params_json)

    test_df = pd.read_csv(args.test_csv)
    graph = build_graph(test_df, args.preprocess_dir, args.rna_ago2_dir, params, os.path.abspath(os.path.join(os.path.dirname(__file__), args.src_root)))

    generator = HinSAGENodeGenerator(graph, params["batch_size"], params["hop_samples"], head_node_type="interaction")
    test_interaction = pd.DataFrame(test_df['efficacy'].values, index=test_df['siRNA'] + "_" + test_df['mRNA'])
    test_gen = generator.flow(test_interaction.index, test_interaction)

    model = tf.keras.models.load_model(args.model_path)
    preds = model.predict(test_gen).squeeze()

    out_df = pd.DataFrame({
        "id": test_df.get("id", pd.Series([f"row_{i}" for i in range(len(test_df))])),
        "label": test_df["efficacy"].astype(float),
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
