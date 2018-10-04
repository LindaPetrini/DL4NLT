import os
from allennlp.modules.elmo import Elmo, batch_to_ids
from dl4nlt.models.embeddings.retrieve_embeddings import download_elmo_weights

data_dir = os.path.join(os.path.dirname(__file__), "../../data")


def create_elmo_embedding(elmo_weights_folder=data_dir):
    """
    Create a pre-trained ELMo embedding PyTorch module. The required embedding data will be downloaded if it is not
    already present.
    :param elmo_weights_folder: Folder in which to find the ELMo embedding data and configuration files
    :return: A PyTorch ELMo embedding module
    """
    elmo_options = os.path.join(elmo_weights_folder, "elmo_options.json")
    elmo_weights = os.path.join(elmo_weights_folder, "elmo_weights.hdf5")

    if not os.path.exists(elmo_options):
        download_elmo_weights()

    elmo = Elmo(elmo_options, elmo_weights, 1, dropout=0)
    return elmo


if __name__ == "__main__":
    elmo = create_elmo_embedding()
    sentences = [['A', 'test', 'sentence', '.'], ['Another', 'test', '.']]
    character_ids = batch_to_ids(sentences)

    embeddings = elmo(character_ids)
    print(embeddings)