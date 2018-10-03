
from gensim.models.doc2vec import Doc2Vec

from dl4nlt.models.w2v_baseline.build_doc2vec import OUTPUT_DIR as DOC2VEC_DIR
from dl4nlt.models.w2v_baseline.svm_doc2vec import OUTPUT_DIR as SVM_DIR

import pickle
import os.path
from dl4nlt import ROOT


from dl4nlt.datasets import load_dataset


from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score

from dl4nlt.models.w2v_baseline import Doc2VecSVR

import numpy as np
import random

DATASET = "global_mispelled"

dataset_path = os.path.join(ROOT, "data", DATASET)
doc2vec_model_path = os.path.join(DOC2VEC_DIR, DATASET)
svr_model_path = os.path.join(SVM_DIR, DATASET)

np.random.seed(42)
random.seed(42)

model = Doc2Vec.load(doc2vec_model_path)


with open(svr_model_path, 'rb') as file:
    svr = pickle.load(file)


trainset, _, testset = load_dataset(dataset_path)


train_pred = []
for i in range(len(trainset)):
    x = trainset[i].essay
    y = trainset[i].y
    train_pred.append(svr.predict(x)[0])
train_pred = np.array(train_pred)

rmse_train = np.sqrt(np.mean((trainset.data['y'] - train_pred.reshape(-1)) ** 2))

print('RMSE on trainingset: ', rmse_train)

c_kappa_train = cohen_kappa_score(2*trainset.data['y'], (2*train_pred).round().astype(int).reshape(-1),
                                  weights="quadratic")
print('Kappa on trainingset: ', c_kappa_train)

spearman_train = spearmanr(trainset.data['y'], train_pred)
print('Spearman r on training set: ', spearman_train)

pearson_train = pearsonr(trainset.data['y'], train_pred.squeeze())
print('Pearson r on training set: ', pearson_train)


test_pred = []
for i in range(len(testset)):
    x = testset[i].essay
    y = testset[i].y
    test_pred.append(svr.predict(x)[0])
test_pred = np.array(test_pred)

rmse_test = np.sqrt(np.mean((testset.data['y'] - test_pred.reshape(-1)) ** 2))

print('RMSE on testset:', rmse_test)

c_kappa_test = cohen_kappa_score(2*testset.data['y'], (2*test_pred).round().astype(int).reshape(-1),
                                 weights="quadratic")
print('Kappa on test set: ', c_kappa_test)

spearman_test = spearmanr(testset.data['y'], test_pred)
print('Spearman r on test set: ', spearman_test)

pearson_test = pearsonr(testset.data['y'], test_pred.squeeze())
print('Pearson r on test set: ', pearson_test)
