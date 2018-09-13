from __future__ import print_function, division
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

#extract the data.zip before running
DATA_FOLDER = "./data/"

class ASAP_Data(Dataset):
    def __init__(self, folder_dataset, essay_set, train=True, valid=False, test=False):
        """
        folder_dataset: folder name where dataset is stored with '/' appended
        essay_set: set of ints, that could be from 1-8 indicating the essay set to be selected
        train: bool to take training set ids
        valid: bool to take validation set ids
        test: bool to take test set ids
        """
        # Open and load tsv file including the whole training data
        data = pd.read_csv(folder_dataset+"data.tsv" ,delimiter="\t", encoding = "ISO-8859-1")
        print(list(data.columns.values)) #file header
        
        # Open the kaggle ids to select the specific dataset: train/valid/test
        split_ids = pd.read_csv(folder_dataset+"kaggle_ids.csv", delimiter=",", encoding = "utf-8")
        datasets_to_include = []
        if train:
            datasets_to_include.append("train")
        if valid:
            datasets_to_include.append("valid")
        if test:
            datasets_to_include.append("test")
        ids_used = split_ids.loc[split_ids['Set'].isin(datasets_to_include)]
        ids_used = list(np.array(ids_used[["ID"]].values.tolist()).flatten())
        
        #select on rows that are in desired dataset and essay set
        data = data.loc[data["essay_set"].isin(essay_set) & data["essay_id"].isin(ids_used)]
        data['y'] = data[['rater1_domain1', 'rater2_domain1']].sum(axis=1)
        
        #if 3rd rater - replace y rating with his instead 
        non_empty_rater3 = list(data.loc[pd.notnull(data['rater3_domain1'])].index.values.tolist())
        for r in non_empty_rater3:
            data.at[r, 'y'] = data.at[r, 'rater3_domain1']
        
        #add preprocessing here to store numbers vectors instead of essays 
        self.data = data[['essay','y']]
        
    
    # Override to give PyTorch access to any item on the dataset
    def __getitem__(self, index):
        x = self.data.at[index, "essay"]
        y = self.data.at[index, "y"]
        return x, y

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.data)

#example 
set1 = ASAP_Data(DATA_FOLDER, [1])
x, y = set1[0]
print(x)
print(y)