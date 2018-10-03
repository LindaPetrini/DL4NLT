
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

spearman_train = spearmanr(trainset.data['y'], train_pred)
print('Spearman r on training set: ', spearman_train)

pearson_train = pearsonr(trainset.data['y'], train_pred.squeeze())
print('Pearson r on training set: ', pearson_train)

denorm_train_pred = denormalize_vec(torch.LongTensor(trainset.data.essay_set), torch.FloatTensor(train_pred).squeeze(), torch.device('cpu')).numpy()

denorm_rmse_train = np.sqrt(np.mean((trainset.data['y_original'] - denorm_train_pred.reshape(-1)) ** 2))

print('RMSE on denormalized trainingset: ', denorm_rmse_train)

denorm_c_kappa_train = cohen_kappa_score(trainset.data['y_original'], denorm_train_pred, weights="quadratic")
print('Kappa on trainingset: ', denorm_c_kappa_train)

denorm_spearman_train = spearmanr(trainset.data['y_original'], denorm_train_pred)
print('Spearman r on denormalized training set: ', denorm_spearman_train)

denorm_pearson_train = pearsonr(trainset.data['y_original'], denorm_train_pred.squeeze())
print('Pearson r on denormalized training set: ', denorm_pearson_train)



test_pred = []
for i in range(len(testset)):
    x = testset[i].essay
    y = testset[i].y
    test_pred.append(svr.predict(x)[0])
test_pred = np.array(test_pred)

rmse_test = np.sqrt(np.mean((testset.data['y'] - test_pred.reshape(-1)) ** 2))

print('RMSE on testset:', rmse_test)

spearman_test = spearmanr(testset.data['y'], test_pred)
print('Spearman r on test set: ', spearman_test)

pearson_test = pearsonr(testset.data['y'], test_pred.squeeze())
print('Pearson r on test set: ', pearson_test)

denorm_test_pred = denormalize_vec(torch.LongTensor(testset.data.essay_set), torch.FloatTensor(test_pred).squeeze(), torch.device('cpu')).numpy()

denorm_rmse_test = np.sqrt(np.mean((testset.data['y_original'] - denorm_test_pred.reshape(-1)) ** 2))

print('RMSE on denormalized testset:', denorm_rmse_test)

denorm_c_kappa_test = cohen_kappa_score(testset.data['y_original'], denorm_test_pred, weights="quadratic")

print('Kappa on denormalized test set: ', denorm_c_kappa_test)

denorm_spearman_test = spearmanr(testset.data['y_original'], denorm_test_pred)
print('Spearman r on denormalized test set: ', denorm_spearman_test)

denorm_pearson_test = pearsonr(testset.data['y_original'], denorm_test_pred.squeeze())
print('Pearson r on denormalized test set: ', denorm_pearson_test)

