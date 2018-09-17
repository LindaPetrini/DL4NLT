
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from dl4nlt.dataloader import ASAP_Data

import argparse

OUTPUT_FILE = "model_doc2vec"
EPOCHS = 20
SIZE = 200
ALPHA = 0.025
WINDOW = 3
WORKERS = 4


def main(essay_set, epochs=EPOCHS, size=SIZE, alpha=ALPHA, window=WINDOW, outfile=OUTPUT_FILE, workers=WORKERS):

    set = ASAP_Data(essay_set)
    
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
    
    model.save(outfile)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--essays', type=str, default=','.join([str(i) for i in range(8)]),
                        help='Comma separated list of essays groups ids (values in [0-7])')
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
    
    main(essay_set=[int(x) for x in FLAGS.essays.split(',')],
         epochs=FLAGS.epochs,
         size=FLAGS.size,
         alpha=FLAGS.alpha,
         window=FLAGS.window,
         workers=FLAGS.workers,
         outfile=FLAGS.output)
