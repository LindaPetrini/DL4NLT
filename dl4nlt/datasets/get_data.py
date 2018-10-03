from __future__ import print_function, division
import os.path

from dl4nlt import ROOT

from dl4nlt.datasets import ASAP_Data
import pickle
import os

DATA_FOLDER = os.path.join(ROOT, "data/")


def get_dataset(name, elmo_formatting=False):
    
    datafolder = os.path.join(DATA_FOLDER, name)

    os.makedirs(datafolder, exist_ok=True)
    
    #Generate and pickle datasets for standard settings
    train = ASAP_Data(list(range(1, 9)),
                      elmo_formatting=elmo_formatting)
    valid = ASAP_Data(list(range(1, 9)), train=False, valid=True, folder_dataset=DATA_FOLDER, dictionary=train.dict,
                      elmo_formatting=elmo_formatting)
    test = ASAP_Data(list(range(1, 9)), train=False, test=True, folder_dataset=DATA_FOLDER, dictionary=train.dict,
                     elmo_formatting=elmo_formatting)

    pickle.dump(train, open(os.path.join(datafolder, 'train.p'), "wb"))
    pickle.dump(valid, open(os.path.join(datafolder, 'valid.p'), "wb"))
    pickle.dump(test, open(os.path.join(datafolder, 'test.p'), "wb"))

    # return train #, valid, test


if __name__ == '__main__':
    # get_dataset('baseline')
    get_dataset('elmo', elmo_formatting=True)
