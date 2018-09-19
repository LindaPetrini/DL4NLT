import wget
import zipfile
import os

elmo_options_url = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
elmo_weights_url = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
glove_weights_url = "http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"

elmo_options_file = "../../data/elmo_options.json"
elmo_weights_file = "../../data/elmo_weights.hdf5"
glove_weights_file = "../../data/glove_weights.dat"


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
    glove_zip_file = "../../data/glove_weights.zip"
    if os.path.exists(glove_zip_file):
        return

    print("Downloading Glove Weights...")
    wget.download(
        "http://nlp.stanford.edu/data/wordvecs/glove.6B.zip",
        out=glove_zip_file,
        bar=wget.bar_thermometer
    )

    with zipfile.ZipFile(glove_zip_file, "r") as zip_ref:
        zip_ref.extractall()

    print("Finished downloading Glove Weights...")


if __name__ == "__main__":
    download_elmo_weights()
    download_glove_weights()