
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
from dl4nlt.models import Doc2VecSVR
import pickle
import os.path
from dl4nlt import ROOT
from dl4nlt.dataloader import ASAP_Data
import numpy as np

doc2vec_model_path = os.path.join(ROOT, "models/w2v_baseline/model_doc2vec")
svr_model_path = os.path.join(ROOT, "models/w2v_baseline/model_svr")


model = Doc2Vec.load(doc2vec_model_path)

with open(svr_model_path, 'rb') as file:
    svr = pickle.load(file)


trainset = ASAP_Data(list(range(1, 9)))


train_pred = []
for i in range(len(trainset)):
    x, y = trainset[i]
    train_pred.append(svr.predict(x)[0])
train_pred = np.array(train_pred)

rmse_train = np.sqrt(np.mean((trainset.data['y'] - train_pred.reshape(-1)) ** 2))

print('RMSE on trainingset: ', rmse_train)

testset = ASAP_Data(list(range(1, 9)), train=False, test=True)
test_pred = []
for i in range(len(testset)):
    x, y = testset[i]
    test_pred.append(svr.predict(x)[0])
test_pred = np.array(test_pred)

rmse_test = np.sqrt(np.mean((testset.data['y'] - test_pred.reshape(-1)) ** 2))

print('RMSE on testset:', rmse_test)