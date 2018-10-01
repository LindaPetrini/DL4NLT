import argparse
import os.path
import pickle

import torch

from torch.utils.data import DataLoader

from torch.nn import MSELoss

from torch.optim import Adam
from torch.optim import SGD
import numpy as np

from dl4nlt import ROOT

OUTPUT_DIR = os.path.join(ROOT, "models/lstm/saved_models")
DATASET_DIR = os.path.join(ROOT, "data/baseline")

from dl4nlt.dataloader import load_dataset
from dl4nlt.dataloader import denormalize
from dl4nlt.dataloader import norm_essay_set
from dl4nlt.datasets import load_dataset
from dl4nlt.models.lstm import CustomLSTM
from dl4nlt.models.lstm import kappa

from tensorboardX import SummaryWriter
from datetime import datetime


##### DEFAULT PARAMS #####

EXP_NAME = 'exp.model'
DROPOUT = 0.3
RNN_TYPE = 'LSTM'
BATCHSIZE = 128
EPOCHS = 10
LR = 4e-3
VALIDATION_BATCHSIZE = 1000


def collate(batch):
    batch = sorted(batch, key=lambda b: -1 * len(b.tokenized))
    lengths = list(map(lambda b: len(b.tokenized), batch))  # Needed for pack_padded in lstm
    
    X = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(b.tokenized).reshape(-1) for b in batch])
    
    s = torch.LongTensor([b.essay_set for b in batch]).reshape(-1)
    
    y = torch.tensor([b.y for b in batch]).reshape(-1)
    
    return X, s, y, lengths


def main(name, dataset, epochs, lr, batchsize, **kwargs):
    def run_epoch(data, epoch, dataset, is_eval=False):
        if is_eval:
            model.eval()
            data_len = valid_len
        else:
            model.train()
            data_len = train_len
        
        loss = 0
        cohen = 0
        for i_batch, batch in enumerate(data):
            x, s, t, l = batch
            x = x.to(device)
            t = t.to(device)
            
            hiddens = model.init_hidden(x.shape[1])
            model.zero_grad()
            
            y = model(x, hiddens, l)[0]
            this_loss = torch.sqrt(criterion(y, t))
            
            # Stuff for Cohen, not working
            # k = []
            # for i in range(y.shape[0]):
            #     y_denorm = int(round(denormalize(s[i].item(), y[i].item())))
            #     t_denorm = int(round(denormalize(s[i].item(), t[i].item())))
            #     max = norm_essay_set[s[i].item()]['max']
            #     min = norm_essay_set[s[i].item()]['min']
            #     k.append(kappa(y_denorm, t_denorm, min_rating=min, max_rating=max))
            # this_cohen = sum(k) / len(k)
            
            # print('\t', i_batch, x.shape, t.shape, y.shape, [h.shape for h in hiddens])
            if not is_eval:
                this_loss.backward()
                optimizer.step()
                writer.add_scalar('Iteration training loss', this_loss.item(), epoch * len(data) + i_batch)
            
            # Weight scores depending on batch size (last batch is smaller)
            loss += this_loss.item() * x.shape[1]
            # cohen += this_cohen * x.shape[1]
        
        # Average over the size of the train/valid data
        loss /= data_len
        cohen /= data_len
        return loss, 0  # TODO cohen
    
    ##############################################
    ## Data, Model and Optimizer initialization ##
    ##############################################
    
    outfile = os.path.join(OUTPUT_DIR, name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    exp_name = 'runs/{} embeddingsFrom_{} dropout_{} batchsize_{} {}'.format(kwargs['rnn_type'],
                                                                             kwargs['embeddings_path'],
                                                                             kwargs['dropout'], batchsize,
                                                                             datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(exp_name)
    
    writer = SummaryWriter(exp_name)
    
    training_set, validation_set, _ = load_dataset(dataset)
    train_len, valid_len = len(training_set), len(validation_set)
    vocab_len = len(training_set.dict)
    
    model = CustomLSTM(vocab_len=vocab_len, **kwargs)
    model.to(device)
    print(model)
    
    training = DataLoader(training_set, batch_size=batchsize, shuffle=True, pin_memory=True, collate_fn=collate)
    validation = DataLoader(validation_set, batch_size=VALIDATION_BATCHSIZE, shuffle=False, pin_memory=True,
                            collate_fn=collate)
    
    criterion = MSELoss()
    # optimizer = Adam(model.parameters(), lr)
    optimizer = SGD(model.parameters(), lr)
    
    print('\n###############################################')
    print('Starting epoch 0 (Random Guessing)')
    loss, cohen = run_epoch(validation, 0, validation_set, is_eval=True)
    print('| Valid Loss: {} |  cohen: {} |\n'.format(loss, cohen))

    train_losses = []
    valid_losses = []
    train_cohen = []
    valid_cohen = []
    

    for e in range(epochs):
        
        print('###############################################')
        print('Starting epoch {}'.format(e + 1))
        
        loss, cohen = run_epoch(training, e, training_set, is_eval=False)
        train_losses.append(loss)
        train_cohen.append(cohen)
        print('| Train Loss: {} |  cohen: {} |'.format(loss, cohen))
        writer.add_scalar('Epoch training loss', loss, e)
        
        loss, cohen = run_epoch(validation, e, validation_set, is_eval=True)
        valid_losses.append(loss)
        valid_cohen.append(cohen)
        print('| Valid Loss: {} |  cohen: {} |'.format(loss, cohen))
        writer.add_scalar('Epoch validation loss', loss, e)
        
        # Save best model so far
        if not len(train_cohen) == 0 and cohen < min(train_cohen):
            with open(outfile, 'wb') as of:
                pickle.dump(model, of)
            print('|\tBest validation accuracy: Model Saved!')
        
        print()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=EXP_NAME,
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
