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

def normalize_row(r):
    return (r['y'] - norm_essay_set[r['essay_set']]['min']
            ) / (norm_essay_set[r['essay_set']]['max'] - norm_essay_set[r['essay_set']]['min'])

def denormalize(essay_set, y):
    return (y + norm_essay_set[essay_set]['min']) * (norm_essay_set[essay_set]['max'] - norm_essay_set[essay_set]['min'])

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
        data['y'] = data[['rater1_domain1', 'rater2_domain1']].mean(axis=1)

        # if 3rd rater - replace y rating with his instead
        non_empty_rater3 = list(data.loc[pd.notnull(data['rater3_domain1'])].index.values.tolist())
        for r in non_empty_rater3:
            data.at[r, 'y'] = data.at[r, 'rater3_domain1'] / 2.0

        print("Number of essays in data: ", len(data['y']))

        #normalize
        print("Normalizing Y axis")
        data['y'] = data.apply(normalize_row, axis=1)

        # add preprocessing here to store numbers vectors instead of essays
        self.data = data[['essay', 'y', 'essay_set']].reset_index(drop=True)
        
        self.dict = Dictionary()
        pre = Preprocessing()
        
        print("\nStarted preprocessing...")
        t_preproc = time.time()
        for i, row in self.data.iterrows():
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


if __name__ == '__main__':
    # example
    set1 = ASAP_Data([1], folder_dataset=DATA_FOLDER)
    x, y = set1[0]
    print(x)
    print(y)