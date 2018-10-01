import os
import pickle
import numpy as np
import bcolz

import torch
import torch.nn as nn

from dl4nlt.models.embeddings.retrieve_embeddings import download_glove_weights, preprocess_glove_weights

data_dir = os.path.join(os.path.dirname(__file__), "../../data/")


def load_weight_matrix(target_vocab):
    print("Loading Glove Weight Matrix...")

    if not os.path.exists(os.path.join(data_dir, "glove_data.dat")):
        download_glove_weights()
        preprocess_glove_weights()

    pretrained_embeddings = bcolz.open(os.path.join(data_dir, "glove_data.dat"))
    pretrained_tokens = pickle.load(open(os.path.join(data_dir, "glove_tokens.pkl"), "rb"))
    pretrained_t2ind = pickle.load(open(os.path.join(data_dir, "glove_t2ind.pkl"), "rb"))

    embedding_size = 50
    weight_matrix = np.zeros((len(target_vocab), embedding_size))
    num_unknown_tokens = 0
    unknown_tokens = set()

    for ind, token in enumerate(target_vocab):
        if token in pretrained_tokens:
            weight_matrix[ind] = pretrained_embeddings[pretrained_t2ind[token]]
        else:
            weight_matrix[ind] = np.random.normal(scale=0.6, size=(embedding_size,))
            unknown_tokens.add(token)
            num_unknown_tokens += 1

    print(f"Loaded Glove Weight Matrix. {num_unknown_tokens} unknown/untrained tokens.")

    return weight_matrix, unknown_tokens


def create_glove_embedding(target_vocab=None, weight_matrix=None):
    """
    Load/create a Pre-Trained Glove Embedding Layer
    """
    if target_vocab is None and weight_matrix is None:
        raise ValueError(
            "Please provide either a target vocabulary or an initialized embedding matrix for the Glove Embedding"
        )
    if weight_matrix is None:
        weight_matrix, _ = load_weight_matrix(target_vocab=target_vocab)

    num_tokens, embedding_dim = weight_matrix.shape

    embedding_layer = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embedding_dim)
    embedding_layer.load_state_dict({"weight": torch.from_numpy(weight_matrix)})

    return embedding_layer


if __name__ == "__main__":
    print(create_glove_embedding(target_vocab=["Hi", "hey"]))
