
from sklearn.svm import SVR

from gensim.models.doc2vec import Doc2Vec

from dl4nlt import ROOT
import os

from dl4nlt.models.w2v_baseline.build_doc2vec import OUTPUT_DIR as DOC2VEC_DIR

OUTPUT_DIR = os.path.join(ROOT, "models/w2v_baseline/saved_models/svm")

import pandas as pd
import argparse

import pickle

from dl4nlt.datasets import load_dataset


class Doc2VecSVR:
    
    def __init__(self, doc2vec, dataset):
        """
        :param doc2vec: either a Doc2Vec model or the path to a saved Doc2Vec model
        :param dataset: the training set
        :return: the trained model
        """
    
        assert isinstance(doc2vec, Doc2Vec) or isinstance(doc2vec, str)
        
        if isinstance(doc2vec, str):
            doc2vec = Doc2Vec.load(doc2vec)
        
        self.doc2vec = doc2vec
        
        set, _, _ = load_dataset(dataset)
    
        y = set.data['y']
        X = set.data['essay'].apply(lambda doc: pd.Series(doc2vec.infer_vector(doc)))
        
        print('Training Set built')
    
        self.svr = SVR(C=1.0, epsilon=0.2)
        self.svr.fit(X, y)
        
        print('SVM trained')
    
    def predict(self, doc):
        
        embedding = self.doc2vec.infer_vector(doc).reshape(1, -1)
        
        return self.svr.predict(embedding), embedding
    
    def save(self, outfile):
        """
        :param outfile: output file where to store the model (default None = model is just returned but not saved)
        """
        with open(outfile, 'wb') as of:
            pickle.dump(self, of)
    
    
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="local_mispelled",
                        help='Dataset name')
    
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Path to the directory where to store the model')

    FLAGS, unparsed = parser.parse_known_args()

    outfile = os.path.join(FLAGS.output_dir, FLAGS.dataset)

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    print(outfile)

    model = Doc2VecSVR(dataset=os.path.join(ROOT, "data/", FLAGS.dataset),
                       doc2vec=os.path.join(DOC2VEC_DIR, FLAGS.dataset))
    
    model.save(outfile=outfile)
    

