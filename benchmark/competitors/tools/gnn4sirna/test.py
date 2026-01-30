#!/usr/bin/env python
import argparse
import os
import sys

import pandas as pd
import tensorflow as tf
import stellargraph as sg
from stellargraph.mapper import HinSAGENodeGenerator
from stellargraph.layer import HinSAGE

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from metrics import format_metrics, regression_metrics, save_metrics


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
    p.add_argument("--test-csv", required=True)
    p.add_argument("--processed-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-csv", default="predictions.csv")
    p.add_argument("--metrics-json", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    graph = load_processed(args.processed_dir)
    test_df = pd.read_csv(args.test_csv)
    test_interaction = pd.DataFrame(test_df["efficacy"].values,
                                    index=test_df["siRNA"] + "_" + test_df["mRNA"])

    generator = HinSAGENodeGenerator(graph, args.batch_size, [8, 4], head_node_type="interaction")
    test_gen = generator.flow(test_interaction.index, test_interaction)

    custom_objects = {"HinSAGE": HinSAGE}
    try:
        from stellargraph.layer import MeanHinAggregator, MeanPoolingAggregator, AttentionalAggregator
        custom_objects.update({
            "MeanHinAggregator": MeanHinAggregator,
            "MeanPoolingAggregator": MeanPoolingAggregator,
            "AttentionalAggregator": AttentionalAggregator,
        })
    except Exception:
        pass
    model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
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
