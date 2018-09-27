import argparse
import os.path
import pickle

import torch

from torch.utils.data import DataLoader

from torch.nn import MSELoss

from torch.optim import Adam
from torch.optim import SGD

from dl4nlt import ROOT
OUTPUT_DIR = os.path.join(ROOT, "models/lstm/saved_models")
DATASET_DIR = os.path.join(ROOT, "data/baseline")

from dl4nlt.datasets import load_dataset
from dl4nlt.models.lstm import CustomLSTM

VALIDATION_BATCHSIZE = 1000

DROPOUT = 0.5
RNN_TYPE = 'LSTM'
BATCHSIZE = 200
EPOCHS = 20
LR = 4e-3


def collate(batch):
    batch = sorted(batch, key=lambda b: -1 * len(b.tokenized))
    
    X = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(b.tokenized).reshape(-1) for b in batch])
    
    s = torch.LongTensor([b.essay_set for b in batch]).reshape(-1)
    
    y = torch.tensor([b.y for b in batch]).reshape(-1)
    
    return X, s, y
        
    
def main(name, dataset, epochs, lr, batchsize, **kwargs):
    
    outfile = os.path.join(OUTPUT_DIR, name)
    
    training, validation, _ = load_dataset(dataset)
    
    vocab_len = len(training.dict)
    
    model = CustomLSTM(vocab_len=vocab_len, **kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    training = DataLoader(training, batch_size=batchsize, shuffle=True, pin_memory=True, collate_fn=collate)
    validation = DataLoader(validation, batch_size=VALIDATION_BATCHSIZE, shuffle=False, pin_memory=True,
                            collate_fn=collate)

    loss = MSELoss()
    # optimizer = Adam(model.parameters(), lr)
    optimizer = SGD(model.parameters(), lr)
    model.to(device)

    train_losses = []
    valid_losses = []

    print('\n###############################################')
    print('Starting epoch 0 (Random Guessing)')

    model.eval()

    valid_loss = 0
    for i_batch, batch in enumerate(validation):
        x, s, t = batch
    
        x = x.to(device)
        t = t.to(device)
    
        hiddens = model.init_hidden(x.shape[1])
    
        y = model(x, hiddens)[0]
    
        l = torch.sqrt(loss(y, t)).item()
    
        valid_loss += l * x.shape[1]

    valid_loss /= len(validation)
    print('| Validation loss: {} |'.format(valid_loss))

    valid_losses.append(valid_loss)

    for e in range(epochs):

        train_loss = 0
        valid_loss = 0
        

        print('###############################################')
        print('Starting epoch {}'.format(e + 1))
        
        model.train()
    
        for i_batch, batch in enumerate(training):
            x, s, t = batch
        
            x = x.to(device)
            t = t.to(device)
        
            hiddens = model.init_hidden(x.shape[1])
            model.zero_grad()
        
            y = model(x, hiddens)[0]
        
            l = torch.sqrt(loss(y, t))

            # print('\t', i_batch, x.shape, t.shape, y.shape, [h.shape for h in hiddens])
            
            l.backward()
            optimizer.step()
        
            train_loss += l.item() * x.shape[1]
    
        train_loss /= len(training)
        train_losses.append(train_loss)
    
        print('| Training loss: {} |'.format(train_loss))

        model.eval()

        for i_batch, batch in enumerate(validation):
            x, s, t = batch
    
            x = x.to(device)
            t = t.to(device)
    
            hiddens = model.init_hidden(x.shape[1])
    
            y = model(x, hiddens)[0]
    
            l = torch.sqrt(loss(y, t)).item()
    
            valid_loss += l * x.shape[1]

        valid_loss /= len(validation)
        print('| Validation loss: {} |'.format(valid_loss))

        if not len(valid_losses) == 0 and valid_loss < min(valid_losses):
            with open(outfile, 'wb') as of:
                pickle.dump(model, of)
            print('|\tModel Saved!')

        valid_losses.append(valid_loss)
    
        # print('Finished epoch {}'.format(e))
        # print('###############################################')
        # print('\n')


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='exp.model',
                        help='Name of the experiment (used for output file name)')
    parser.add_argument('--dataset', type=str, default=DATASET_DIR,
                        help='Path to the folder containg the dataset')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=LR,
                        help='Value of learning rate')
    parser.add_argument('--batchsize', type=int, default=BATCHSIZE,
                        help='The batch-size for the training')
    
    parser.add_argument('--emb_size', type=int, default=200,
                        help='Size of the embeddings')
    parser.add_argument('--n_hidden_units', type=int, default=100,
                        help='Number of units per hidden layer')
    parser.add_argument('--n_hidden_layers', type=int, default=1,
                        help='Number of hidden layers')
    parser.add_argument('--dropout', type=float, default=DROPOUT,
                        help='Dropout probability')
    parser.add_argument('--rnn_type', type=str, default=RNN_TYPE,
                        help='Type of RNN model to use')
    parser.add_argument('--embeddings_path', type=str, default=None,
                        help='Path to the file containing the pre-trained embeddings')
    
    FLAGS = parser.parse_args()
    main(**vars(FLAGS))

