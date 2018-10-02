import os.path
import torch


from dl4nlt import ROOT

EMB_PATH = os.path.join(ROOT, "models/sswe/saved_models", "exp.model")
DATASET_DIR = os.path.join(ROOT, "data/baseline")

from dl4nlt.datasets import load_dataset

from dl4nlt.models.sswe import SSWEModel

training, _, _ = load_dataset(DATASET_DIR)

# model = some_nice_wrapper(EMB_PATH)
model = SSWEModel(vocab_len=len(training.dict), context_size=9, emb_size=200, n_hidden_units=100)


embeddings = model.embeddings


words = ['computer', 'people', 'laptop']

for w in words:
    print(w)
    
    w1 = training.dict.word2idx[w]
    w2 = training.dict.word2idx['_' + w]
    
    e1 = embeddings(torch.tensor(w1))
    e2 = embeddings(torch.tensor(w2))
    
    print('\tCosineSimilarity with the mispelled version =', (e1@e2).item())

for i in range(len(words)-1):
    for j in range(i+1, len(words)):
        word1 = words[i]
        word2 = words[j]
        
        print(word1, '-', word2)

        w1 = training.dict.word2idx[word1]
        w2 = training.dict.word2idx[word2]

        e1 = embeddings(torch.tensor(w1))
        e2 = embeddings(torch.tensor(w2))

        print('\tCosineSimilarity =', (e1 @ e2).item())
        
    
    


