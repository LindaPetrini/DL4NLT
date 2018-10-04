import os.path
import torch

from dl4nlt import ROOT
from dl4nlt.models.lstm import CustomLSTM

from dl4nlt.datasets import load_dataset

################################################################

# SET THIS PARAMETER
EXP_NAME = '(LSTM;sswe;local_mispelled;0.0001;128;0.4;1;1)'

DATASET = 'local_mispelled'

MODEL_PATH = os.path.join(ROOT, "models/lstm/saved_data", DATASET, EXP_NAME, "model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


training, _, _ = load_dataset(os.path.join(ROOT, 'data', DATASET))

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = CustomLSTM(device=device, **checkpoint['params'])
model.load_state_dict(checkpoint['state_dict'])

embeddings = model.encoder



# words = ['computer', 'people', 'laptop']
# words = ['computer', '_computer', 'laptop', '_laptop']
# words = ['girl', '_girl', 'woman', '_woman', 'man', '_man']
words = ['student', 'teacher', '_student', '_teacher']

# words = ['', '', '',
        #  '']  # TODO Find some good one where boy = girl, _boy ~ _girl, boy ~ _boy, but _boy != girl (both semantic and correctness are different)
# words = ['', '', '', '']
# words = ['', '', '', '']
# words = ['', '', '', '']
# words = ['', '', '', '']

# for w in words:
#     print(w)
#
#     w1 = training.dict.word2idx[w]
#     w2 = training.dict.word2idx['_' + w]
#
#     e1 = embeddings(torch.tensor(w1))
#     e2 = embeddings(torch.tensor(w2))
#
#     print('\tCosineSimilarity with the mispelled version =', (e1@e2).item())

for i in range(len(words) - 1):
    for j in range(i + 1, len(words)):
        word1 = words[i]
        word2 = words[j]
        
        print(word1, '-', word2)
        
        w1 = training.dict.word2idx[word1]
        w2 = training.dict.word2idx[word2]
        
        e1 = embeddings(torch.tensor(w1).to(device))
        e2 = embeddings(torch.tensor(w2).to(device))

        print("\t CosineSimilarity using torch = ", torch.nn.functional.cosine_similarity(e1, e2, dim=0, eps=1e-8).item())

        print('\t Dot Product =', (e1 @ e2).item())





