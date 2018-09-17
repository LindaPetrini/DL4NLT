from __future__ import print_function, division

import numpy as np
import pandas as pd

import os.path

from torch.utils.data.dataset import Dataset

from .preprocessing import Preprocessing, Dictionary
import time

from dl4nlt import ROOT

# extract the data.zip before running
DATA_FOLDER = os.path.join(ROOT, "data/")


class ASAP_Data(Dataset):
    def __init__(self, essay_set, folder_dataset=DATA_FOLDER, train=True, valid=False, test=False):
        """
        essay_set: set of ints, that could be from 1-8 indicating the essay set to be selected
        folder_dataset: folder name where dataset is stored
        train: bool to take training set ids
        valid: bool to take validation set ids
        test: bool to take test set ids
        """
        # Open and load tsv file including the whole training data
        data = pd.read_csv(os.path.join(folder_dataset,"data.tsv"), delimiter="\t", encoding="ISO-8859-1")
        print(list(data.columns.values))  # file header
        
        # Open the kaggle ids to select the specific dataset: train/valid/test
        split_ids = pd.read_csv(folder_dataset + "kaggle_ids.csv", delimiter=",", encoding="utf-8")
        datasets_to_include = []
        if train:
            datasets_to_include.append("train")
        if valid:
            datasets_to_include.append("valid")
        if test:
            datasets_to_include.append("test")
        ids_used = split_ids.loc[split_ids['Set'].isin(datasets_to_include)]
        ids_used = list(np.array(ids_used[["ID"]].values.tolist()).flatten())
        
        # select on rows that are in desired dataset and essay set
        data = data.loc[data["essay_set"].isin(essay_set) & data["essay_id"].isin(ids_used)]
        data['y'] = data[['rater1_domain1', 'rater2_domain1']].mean(axis=1)
        
        # if 3rd rater - replace y rating with his instead
        non_empty_rater3 = list(data.loc[pd.notnull(data['rater3_domain1'])].index.values.tolist())
        for r in non_empty_rater3:
            data.at[r, 'y'] = data.at[r, 'rater3_domain1'] / 2.0
        
        # add preprocessing here to store numbers vectors instead of essays
        self.data = data[['essay', 'y']]
        self.dict = Dictionary()
        pre = Preprocessing()
        
        print("\nStarted preprocessing...")
        t_preproc = time.time()
        for i, row in data.iterrows():
            x = self.data.at[i, "essay"]
            preprocessed_essay = pre.preprocess_essay(x)
            for w in preprocessed_essay:
                self.dict.add_word(w)
            self.data.at[i, "essay"] = preprocessed_essay
        self.longest_essay = pre.max_len
        print("Preprocessing completed! Took {:.2f} s\n".format(time.time()-t_preproc))
    
    # Override to give PyTorch access to any item on the dataset
    def __getitem__(self, index):
        x = self.data.at[index, "essay"]
        y = self.data.at[index, "y"]
        return x, y
    
    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.data)


# example
set1 = ASAP_Data([1], folder_dataset=DATA_FOLDER)
x, y = set1[0]
print(x)
print(y)
