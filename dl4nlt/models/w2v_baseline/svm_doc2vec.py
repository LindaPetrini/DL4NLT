
from sklearn.svm import SVR

from gensim.models.doc2vec import Doc2Vec
from dl4nlt.dataloader import ASAP_Data

from .build_doc2vec import OUTPUT_FILE as DOC2VEC_MODEL

import pandas as pd
import argparse

import pickle

OUTPUT_FILE = "model_svr"


class Doc2VecSVR:
    
    def __init__(self, doc2vec, essay_set=list(range(8))):
        """
        :param doc2vec: either a Doc2Vec model or the path to a saved Doc2Vec model
        :param essay_set: list of ids of the essays sub-categories to use
        :return: the trained model
        """
    
        assert isinstance(doc2vec, Doc2Vec) or isinstance(doc2vec, str)
        
        if isinstance(doc2vec, str):
            doc2vec = Doc2Vec.load(doc2vec)
        
        self.doc2vec = doc2vec
        
        set = ASAP_Data(essay_set)
    
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
    parser.add_argument('--essays', type=str, default=','.join([str(i) for i in range(8)]),
                        help='Comma separated list of essays groups ids (values in [0-7])')
    parser.add_argument('--doc2vec', type=str, default=DOC2VEC_MODEL,
                        help='Path to the already trained Doc2Vec model')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE,
                        help='File where to store the model')
    FLAGS, unparsed = parser.parse_known_args()
    
    model = Doc2VecSVR(essay_set=[int(x) for x in FLAGS.essays.split(',')], doc2vec=FLAGS.doc2vec)
    
    model.save(outfile=FLAGS.output)
