"""Inductive graph views: training neighbors only, isolated evaluation queries."""
import pandas as pd
import stellargraph as sg


def training_graph(graph, interaction_ids):
    nodes = set(interaction_ids)
    for interaction in interaction_ids:
        nodes.update(graph.neighbors(interaction))
    return graph.subgraph(sorted(nodes))


def isolated_queries(graph, interaction_ids):
    """Clone sequence nodes per query so evaluation samples cannot exchange features.

    Batching these disconnected three-node components is equivalent to building
    one graph per query, with no train or other held-out interactions attached.
    """
    ids = list(interaction_ids)
    sequence_rows = {"siRNA": [], "mRNA": []}
    sequence_ids = {"siRNA": [], "mRNA": []}
    edges = []
    for interaction in ids:
        neighbors = {"siRNA": [], "mRNA": []}
        for node in graph.neighbors(interaction):
            neighbors[graph.node_type(node)].append(node)
        if any(len(nodes) != 1 for nodes in neighbors.values()):
            raise ValueError("Each interaction must have exactly one siRNA and one mRNA neighbor")
        for kind, nodes in neighbors.items():
            query_node = interaction + "::query_" + kind
            sequence_ids[kind].append(query_node)
            sequence_rows[kind].append(graph.node_features(nodes, node_type=kind)[0])
            edges.append({"source": interaction, "target": query_node})
    features = {kind: pd.DataFrame(sequence_rows[kind], index=sequence_ids[kind]) for kind in sequence_rows}
    features["interaction"] = pd.DataFrame(graph.node_features(ids, node_type="interaction"), index=ids)
    return sg.StellarGraph(features, edges=pd.DataFrame(edges), source_column="source", target_column="target")
