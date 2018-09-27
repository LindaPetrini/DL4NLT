from __future__ import print_function, division

import numpy as np
import pandas as pd

import os.path

from torch.utils.data.dataset import Dataset

import time

from dl4nlt import ROOT

from dl4nlt.datasets import ASAP_Data

# extract the data.zip before running
DATA_FOLDER = os.path.join(ROOT, "models/sswe/data/")


class ContextDateset(Dataset):
    
    def __init__(self, dataset: ASAP_Data, context_size, error_rate):
        
        essays = dataset.data
        
        ngrams = []
        
        for i in range(len(essays)):
            tokenized = essays.iloc[i].tokenized
            y = essays.iloc[i].y
            
            for j in range(len(tokenized) - context_size):
                ngrams.append((tokenized[j: j+context_size], y))
        
        self.data = pd.DataFrame(ngrams, columns=['ngram', 'y'])
        self.dict = dataset.dict
    
    # Override to give PyTorch access to any item on the dataset
    def __getitem__(self, index):
        return self.data.iloc[index]
    
    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.data)
