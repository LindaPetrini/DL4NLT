from allennlp.modules.elmo import Elmo, batch_to_ids

DEFAULT_ELMO_LOCATION = "../../data"


def create_elmo_embedding(elmo_weights_folder=DEFAULT_ELMO_LOCATION):
    elmo_options = f"{elmo_weights_folder}/elmo_options.json"
    elmo_weights = f"{elmo_weights_folder}/elmo_weights.hdf5"

    elmo = Elmo(elmo_options, elmo_weights, 2, dropout=0)
    return elmo


if __name__ == "__main__":
    elmo = create_elmo_embedding()
    sentences = [['A', 'test', 'sentence', '.'], ['Another', 'test', '.']]
    character_ids = batch_to_ids(sentences)

    embeddings = elmo(character_ids)
    print(embeddings)