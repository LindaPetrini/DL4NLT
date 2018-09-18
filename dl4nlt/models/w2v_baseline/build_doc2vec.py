
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from dl4nlt.dataloader import ASAP_Data
import argparse
import os.path

from dl4nlt import ROOT


OUTPUT_FILE = os.path.join(ROOT, "models/w2v_baseline/model_doc2vec")
EPOCHS = 20
SIZE = 200
ALPHA = 0.025
WINDOW = 3
WORKERS = 4


def train_doc2vec(essay_set=list(range(1, 9)), epochs=EPOCHS, size=SIZE, alpha=ALPHA, window=WINDOW, outfile=None, workers=WORKERS):
    """
    :param essay_set: list of ids of the essays sub-categories to use
    :param epochs: number of epochs to train the Doc2Vec model
    :param size: size of the embeddings for Doc2Vec
    :param alpha: learning rate for Doc2Vec
    :param window: size of window for Doc2Vec
    :param outfile: output file where to store the model (default None = model is just returned but not saved)
    :param workers: number of parallel processes to run
    :return: the trained model
    """
    
    set = ASAP_Data(essay_set, train=True, valid=False, test=False)
    
    documents = [TaggedDocument(set[i][0], [i]) for i in range(len(set))]
    
    # dm = 1 for 'Distributed Memory' (PV-DM); dm=0 uses 'Distributed Bag-of-Words' which does not preserve words order
    model = Doc2Vec(vector_size=size, dm=1, alpha=alpha, window=window, min_count=1, workers=workers)

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
    parser.add_argument('--essays', type=str, default=','.join([str(i) for i in range(1, 9)]),
                        help='Comma separated list of essays groups ids (values in [1-8])')
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
    parser.add_argument('--output', type=str, default=OUTPUT_FILE,
                        help='File where to store the model')
    FLAGS, unparsed = parser.parse_known_args()
    
    train_doc2vec(essay_set=[int(x) for x in FLAGS.essays.split(',')],
                  epochs=FLAGS.epochs,
                  size=FLAGS.size,
                  alpha=FLAGS.alpha,
                  window=FLAGS.window,
                  workers=FLAGS.workers,
                  outfile=FLAGS.output)
