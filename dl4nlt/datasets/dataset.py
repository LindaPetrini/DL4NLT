from __future__ import print_function, division

import numpy as np
import pandas as pd

import os.path

import torch
from torch.utils.data.dataset import Dataset

from dl4nlt.datasets.preprocessing import Preprocessing, Dictionary
import time

from dl4nlt import ROOT

# extract the data.zip before running
DATA_FOLDER = os.path.join(ROOT, "data/")

norm_essay_set = {
    1: {'max': 6, 'min': 1},
    2: {'max': 6, 'min': 1},
    3: {'max': 3, 'min': 0},
    4: {'max': 3, 'min': 0},
    5: {'max': 4, 'min': 0},
    6: {'max': 4, 'min': 0},
    7: {'max': 12, 'min': 0},
    8: {'max': 30, 'min': 5}
}

norm_essay_set_tens = torch.FloatTensor([[x['max'], x['min']] for k, x in sorted(norm_essay_set.items())])


def normalize_row(r):
    return (r['y_original'] - norm_essay_set[r['essay_set']]['min']
            ) / (norm_essay_set[r['essay_set']]['max'] - norm_essay_set[r['essay_set']]['min'])


def denormalize(essay_set, y):
    return (y + norm_essay_set[essay_set]['min']) * (
            norm_essay_set[essay_set]['max'] - norm_essay_set[essay_set]['min'])


def denormalize_vec(essay_set, y, device):
    assert essay_set.shape == y.shape
    assert essay_set.max().item() <= 8
    assert essay_set.min().item() >= 1
    
    norm_essay_set_tens_device = norm_essay_set_tens.to(device)
    
    essay_set = essay_set - 1
    a = norm_essay_set_tens_device[essay_set, 1]
    b = norm_essay_set_tens_device[essay_set, 0]
    return (a + y * (b - a)).round().type(torch.LongTensor).to(device)


class ASAP_Data(Dataset):
    def __init__(self, essay_set, folder_dataset=DATA_FOLDER,
                 train=True, valid=False, test=False, dictionary=None, global_misspelled_token=False,
                 elmo_formatting=False):
        """
        essay_set: set of ints, that could be from 1-8 indicating the essay set to be selected
        folder_dataset: folder name where dataset is stored
        train: bool to take training set ids
        valid: bool to take validation set ids
        test: bool to take test set ids
        """
        # Open and load tsv file including the whole training data
        data = pd.read_csv(os.path.join(folder_dataset, "data.tsv"), delimiter="\t", encoding="ISO-8859-1")
        # file header
        
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
        data['y_original'] = data[['rater1_domain1', 'rater2_domain1']].mean(axis=1)
        
        # if 3rd rater - replace y rating with his instead
        non_empty_rater3 = list(data.loc[pd.notnull(data['rater3_domain1'])].index.values.tolist())
        for r in non_empty_rater3:
            data.at[r, 'y_original'] = data.at[r, 'rater3_domain1'] / 2.0
        
        print("Number of essays in data: ", len(data['y_original']))
        
        # normalize
        print("Normalizing Y axis")
        data['y'] = data.apply(normalize_row, axis=1)
        
        # add preprocessing here to store numbers vectors instead of essays
        self.data = data[['essay', 'y', 'essay_set', 'y_original']].reset_index(drop=True)
        
        if not elmo_formatting:
            self.dict = dictionary
            self.preprocess_essays(global_misspelled_token)
            
            self.data.loc[:, 'tokenized'] = self.data.essay.apply(lambda e: [self.dict.word2idx.get(w, 1) for w in e])
    
    # Override to give PyTorch access to any item on the dataset
    def __getitem__(self, index):
        return self.data.iloc[index]
    
    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.data)
    
    def preprocess_essays(self, global_misspelled_token):
        build_dict = False
        if self.dict is None:
            self.dict = Dictionary()
            build_dict = True
            print("Building Dictionary for Set of Essays of length: ", len(self.data))
        
        pre = Preprocessing(global_misspelled_token)
        
        print("\nStarted preprocessing...")
        t_preproc = time.time()
        for i, row in self.data.iterrows():
            essay = self.data.at[i, "essay"]
            preprocessed_essay = pre.preprocess_essay(essay)
            if build_dict:
                for w in preprocessed_essay:
                    self.dict.add_word(w)
            self.data.at[i, "essay"] = preprocessed_essay
        
        print("Preprocessing completed! Took {:.2f} s\n".format(time.time() - t_preproc))
        self.longest_essay = pre.max_len


if __name__ == '__main__':
    # example
    set1 = ASAP_Data([1], folder_dataset=DATA_FOLDER)
    x, y = set1[0]
    print(x)
    print(y)
