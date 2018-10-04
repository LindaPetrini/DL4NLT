
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from dl4nlt.datasets import ASAP_Data
import argparse
import os.path
import numpy as np
import random

from dl4nlt import ROOT

from dl4nlt.datasets import load_dataset

OUTPUT_DIR = os.path.join(ROOT, "models/w2v_baseline/saved_models/doc2vec")

EPOCHS = 20
SIZE = 200
ALPHA = 0.025
WINDOW = 3
WORKERS = 1

np.random.seed(42)
random.seed(42)


def train_doc2vec(dataset, epochs=EPOCHS, size=SIZE, alpha=ALPHA, window=WINDOW, outfile=None, workers=WORKERS):
    """
    :param dataset: path to the dataset
    :param epochs: number of epochs to train the Doc2Vec model
    :param size: size of the embeddings for Doc2Vec
    :param alpha: learning rate for Doc2Vec
    :param window: size of window for Doc2Vec
    :param outfile: output file where to store the model (default None = model is just returned but not saved)
    :param workers: number of parallel processes to run
    :return: the trained model
    """
    
    if outfile is not None:
        outdir = os.path.dirname(outfile)
        os.makedirs(outdir, exist_ok=True)
    
    set, _, _ = load_dataset(dataset)
    
    documents = [TaggedDocument(set[i].essay, [i]) for i in range(len(set))]
    
    # dm = 1 for 'Distributed Memory' (PV-DM); dm=0 uses 'Distributed Bag-of-Words' which does not preserve words order
    model = Doc2Vec(vector_size=size, dm=1, alpha=alpha, window=window, min_count=1, workers=workers, seed=42)

    model.build_vocab(documents)

    for epoch in range(epochs):
        print('iteration {0}'.format(epoch))
        model.train(documents,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    
    if outfile is not None:
        model.save(outfile)
    
    return model


if __name__ == '__main__':
    # Command line arguments

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="local_mispelled",
                        help='Dataset name')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs for training')
    parser.add_argument('--size', type=int, default=SIZE,
                        help='Size of the embeddings')
    parser.add_argument('--alpha', type=float, default=ALPHA,
                        help='Value of parameter Alpha of Doc2Vec')
    parser.add_argument('--window', type=int, default=WINDOW,
                        help='Size of the window for the Doc2Vec model')
    parser.add_argument('--workers', type=int, default=WORKERS,
                        help='Number of parallel processes to run')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Directory where to store the model')
    FLAGS, unparsed = parser.parse_known_args()
    
    train_doc2vec(dataset=os.path.join(ROOT, "data/", FLAGS.dataset),
                  epochs=FLAGS.epochs,
                  size=FLAGS.size,
                  alpha=FLAGS.alpha,
                  window=FLAGS.window,
                  workers=FLAGS.workers,
                  outfile=os.path.join(FLAGS.output_dir, FLAGS.dataset))