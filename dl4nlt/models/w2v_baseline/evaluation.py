
from gensim.models.doc2vec import Doc2Vec

from dl4nlt.models.w2v_baseline.build_doc2vec import OUTPUT_DIR as DOC2VEC_DIR
from dl4nlt.models.w2v_baseline.svm_doc2vec import OUTPUT_DIR as SVM_DIR

import pickle
import os.path
from dl4nlt import ROOT


from dl4nlt.datasets import load_dataset
from dl4nlt.datasets import denormalize_vec

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score

from dl4nlt.models.w2v_baseline import Doc2VecSVR

import torch
import numpy as np
import random

import argparse

np.random.seed(42)
random.seed(42)


def evaluate(svr, data):
    pred = []
    targs = []
    for i in range(len(data)):
        x = data[i].essay
        y = data[i].y_original
        pred.append(svr.predict(x)[0])
        targs.append(y)
        
    pred = np.array(pred).squeeze()
    targs = np.array(targs).squeeze()
    
    rmse = np.sqrt(np.mean((targs - pred) ** 2))
    
    print('RMSE: ', rmse)
    
    spearman = spearmanr(pred, targs)
    print('Spearman r: ', spearman)
    
    pearson = pearsonr(pred, targs)
    print('Pearson r: ', pearson)
    
    # pred = denormalize_vec(torch.LongTensor(data.data.essay_set), torch.FloatTensor(pred), device='cpu').numpy()
    # targs = denormalize_vec(torch.LongTensor(data.data.essay_set), torch.FloatTensor(targs), device='cpu').numpy()

    kappa = cohen_kappa_score(targs.astype(int), pred.round().astype(int), weights="quadratic")
    print('Kappa: ', kappa)

    # rmse = np.sqrt(np.mean((targs - pred) ** 2))
    #
    # print('Denorm RMSE: ', rmse)
    #
    # spearman = spearmanr(pred, targs)
    # print('Denorm Spearman r: ', spearman)
    #
    # pearson = pearsonr(pred, targs)
    # print('Denorm Pearson r: ', pearson)


def main(dataset):
    
    dataset_path = os.path.join(ROOT, "data", dataset)
    svr_model_path = os.path.join(SVM_DIR, dataset)
    
    with open(svr_model_path, 'rb') as file:
        svr = pickle.load(file)
    
    trainset, _, testset = load_dataset(dataset_path)

    print('####DATASET####')
    
    print(dataset)
    
    print('#########TRAINING#########')
    
    evaluate(svr, trainset)
    
    print('#########TESTING#########')
    
    evaluate(svr, testset)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="global_mispelled",
                        help='Dataset name')
    
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS.dataset)


