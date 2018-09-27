import wget
import zipfile
import os
import pickle
import bcolz
import numpy as np

elmo_options_url = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
elmo_weights_url = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
glove_weights_url = "http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"

data_dir = os.path.join(os.path.dirname(__file__), "../../data/")
elmo_options_file = os.path.join(data_dir, "elmo_options.json")
elmo_weights_file = os.path.join(data_dir, "elmo_weights.hdf5")
glove_weights_file = os.path.join(data_dir, "glove_weights.dat")


def download_elmo_weights():
    if os.path.exists(elmo_options_file):
        return

    print("Downloading Elmo Weights...")
    wget.download(
        elmo_options_url,
        out=elmo_options_file
    )

    wget.download(
        elmo_weights_url,
        out=elmo_weights_file
    )
    print("Finished downloading Elmo Weights...")


def download_glove_weights():
    glove_zip_file = os.path.join(data_dir, "glove_weights.zip")
    if not os.path.exists(glove_zip_file):
        print("Downloading Glove Weights...")
        wget.download(
            "http://nlp.stanford.edu/data/wordvecs/glove.6B.zip",
            out=glove_zip_file,
            bar=wget.bar_thermometer
        )
        print("Finished downloading Glove Weights...")

    print("Extracting Glove Weights...")
    with zipfile.ZipFile(glove_zip_file, "r") as zip_ref:
        zip_ref.extractall(path=data_dir)
    print("Finished extracting Glove Weights")


def preprocess_glove_weights():
    """
    Preprocess the Glove weights for use in Pytorch embedding

    https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

    """
    print("Preparing Glove Weights")
    tokens = []
    index = 0
    t2ind = {}
    embeddings = bcolz.carray(np.zeros(1), rootdir=os.path.join(data_dir, "glove_data.dat"), mode="w")

    with open(os.path.join(data_dir, "glove.6B.50d.txt"), "rb") as weights_file:
        for line in weights_file:
            line_vals = line.decode().split()
            token = line_vals[0]
            tokens.append(token)

            t2ind[token] = index
            index += 1

            embedding = np.array(line_vals[1:]).astype(np.float)
            embeddings.append(embedding)

    embeddings = bcolz.carray(embeddings[1:].reshape((-1, 50)),
                              rootdir=os.path.join(data_dir, "glove_data.dat"),
                              mode='w')
    embeddings.flush()
    pickle.dump(tokens, open(os.path.join(data_dir, "glove_tokens.pkl"), "wb"))
    pickle.dump(t2ind, open(os.path.join(data_dir, "glove_t2ind.pkl"), "wb"))
    print("Finished Preparing Glove Weights")


if __name__ == "__main__":
    download_elmo_weights()
    download_glove_weights()
    preprocess_glove_weights()
