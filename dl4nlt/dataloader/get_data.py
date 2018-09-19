from __future__ import print_function, division
import os.path

from dl4nlt import ROOT

from dl4nlt.dataloader import ASAP_Data
import pickle
import os

DATA_FOLDER = os.path.join(ROOT, "data/")

def get_dataset(name):
    #Generate and pickle datasets for standard settings
    train = ASAP_Data(list(range(1, 9)))
    valid = ASAP_Data(list(range(1, 9)), train=False, valid=True, folder_dataset=DATA_FOLDER, dictionary=train.dict)
    test = ASAP_Data(list(range(1, 9)), train=False, test=True, folder_dataset=DATA_FOLDER, dictionary=train.dict)

    pickle.dump(train, open(os.path.join(DATA_FOLDER, name+'train.p'), "wb"))
    pickle.dump(valid, open(os.path.join(DATA_FOLDER, name+'valid.p'), "wb"))
    pickle.dump(test, open(os.path.join(DATA_FOLDER, name+'test.p'), "wb"))

    return train #, valid, test
if __name__ == '__main__':
    get_dataset('baseline_')