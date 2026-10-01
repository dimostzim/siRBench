import torch


def pad_mrna_embedding(embedding, sequence_length, left_padding, right_padding):
    """Preserve BOS at index zero and any antisense suffix after the mRNA."""
    return torch.cat([
        embedding[:1],
        embedding.new_zeros((left_padding, embedding.shape[1])),
        embedding[1:1 + sequence_length],
        embedding.new_zeros((right_padding, embedding.shape[1])),
        embedding[1 + sequence_length:],
    ], dim=0)
