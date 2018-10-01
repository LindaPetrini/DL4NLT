
import os.path
import pickle


def load_dataset(path):
    
    with open(os.path.join(path, 'train.p'), 'rb') as file:
        train = pickle.load(file)
    
    with open(os.path.join(path, 'valid.p'), 'rb') as file:
        valid = pickle.load(file)
    
    with open(os.path.join(path, 'test.p'), 'rb') as file:
        test = pickle.load(file)
    
    return train, valid, test