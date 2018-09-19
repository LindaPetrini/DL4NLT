import argparse
import os.path
import pickle

import torch
import numpy as np
import random

np.random.seed(42)
random.seed(42)

from .lstm import CustomLSTM

from torch.utils.data import DataLoader

from dl4nlt import ROOT

OUTPUT_DIR = os.path.join(ROOT, "models/lstm/saved_models")

from dl4nlt.dataloader import load_dataset

BATCHSIZE = 200
EPOCHS = 20
LR = 2e-3


def collate(batch):
    batch = sorted(batch, key=lambda b: -1 * len(b.tokenized))
    
    X = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(b.tokenized).reshape(-1, 1) for b in batch])
    
    s = torch.LongTensor([b.essay_set for b in batch]).reshape(1, -1, 1)
    
    y = torch.tensor([b.y for b in batch]).reshape(1, -1, 1)
    
    return X, s, y


def train(model, dataset, epochs, lr, batchsize):
    
    train, valid, test = dataset
    
    dataloader = DataLoader(train, batch_size=batchsize, shuffle=True, collate_fn=collate)
    
    for e in range(epochs):
        print('###############################################')
        print('Starting epoch {}'.format(e))
        print('###############################################')
        for i_batch, batch in enumerate(dataloader):
            pass

        print('###############################################')
        print('Finished epoch {}'.format(e))
        print('###############################################')
        print('\n')
        
    
def main(name, dataset, epochs, lr, batchsize, **kwargs):
    
    train, valid, test = load_dataset(dataset)
    
    vocab_len = None
    
    model = CustomLSTM(vocab_len=vocab_len, **kwargs)
    
    dataset = None
    
    train(model, (train, valid, test), epochs, lr, batchsize)
    
    outfile = os.path.join(OUTPUT_DIR, name)
    
    with open(outfile, 'wb') as of:
        pickle.dump(model, of)
    
    return model


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,
                        help='Name of the experiment (used for output file name)')
    parser.add_argument('--dataset', type=str,
                        help='Path tot he folder containg the dataset')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=LR,
                        help='Value of learning rate')
    parser.add_argument('--batchsize', type=int, default=BATCHSIZE,
                        help='The batch-size for the training')
    
    parser.add_argument('--emb_size', type=int,
                        help='Size of the embeddings')
    parser.add_argument('--n_hidden_units', type=int,
                        help='Number of units per hidden layer')
    parser.add_argument('--n_hidden_layers', type=int,
                        help='Number of hidden layers')
    parser.add_argument('--dropout', type=float,
                        help='Dropout probability')
    parser.add_argument('--rnn_type', type=str,
                        help='Type of RNN model to use')
    parser.add_argument('--embeddings_path', type=str,
                        help='Path to the file containing the pre-trained embeddings')
    
    FLAGS = parser.parse_args()
    main(**vars(FLAGS))

