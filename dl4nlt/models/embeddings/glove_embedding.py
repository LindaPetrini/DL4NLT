import os
import pickle
import numpy as np
import bcolz

import torch
import torch.nn as nn

from dl4nlt.models.embeddings.retrieve_embeddings import download_glove_weights, preprocess_glove_weights

data_dir = os.path.join(os.path.dirname(__file__), "../../data/")


def load_weight_matrix(target_vocab_to_idx, emb_size):
    print("Loading Glove Weight Matrix...")

    if not os.path.exists(os.path.join(data_dir, f"glove_data{emb_size}.dat")):
        download_glove_weights()
        preprocess_glove_weights()

    pretrained_embeddings = bcolz.open(os.path.join(data_dir, f"glove_data{emb_size}.dat"))
    pretrained_tokens = pickle.load(open(os.path.join(data_dir, f"glove_tokens{emb_size}.pkl"), "rb"))
    pretrained_t2ind = pickle.load(open(os.path.join(data_dir, f"glove_t2ind{emb_size}.pkl"), "rb"))

    embedding_size = emb_size
    weight_matrix = np.zeros((len(target_vocab_to_idx), embedding_size))
    num_unknown_tokens = 0
    unknown_tokens = set()

    for token, id in target_vocab_to_idx.items():
        if token in pretrained_tokens:
            weight_matrix[id] = pretrained_embeddings[pretrained_t2ind[token]]
        else:
            weight_matrix[id] = np.random.normal(scale=0.6, size=(embedding_size,))
            unknown_tokens.add(token)
            num_unknown_tokens += 1

    print(f"Loaded Glove Weight Matrix. {num_unknown_tokens} unknown/untrained tokens.")

    return weight_matrix, unknown_tokens


def create_glove_embedding(target_vocab_to_idx=None, weight_matrix=None, emb_size=200):
    """
    Load/create a Pre-Trained Glove Embedding Layer
    """
    if target_vocab_to_idx is None and weight_matrix is None:
        raise ValueError(
            "Please provide either a target vocabulary or an initialized embedding matrix for the Glove Embedding"
        )
    if weight_matrix is None:
        weight_matrix, _ = load_weight_matrix(target_vocab_to_idx=target_vocab_to_idx, emb_size=emb_size)

    num_tokens, embedding_dim = weight_matrix.shape

    embedding_layer = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embedding_dim)
    embedding_layer.load_state_dict({"weight": torch.from_numpy(weight_matrix)})

    return embedding_layer


if __name__ == "__main__":
    print(create_glove_embedding(target_vocab=["Hi", "hey"]))
