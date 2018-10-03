import argparse
import os.path
import pickle
import csv
import time

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim import SGD

import numpy as np

from allennlp.modules.elmo import batch_to_ids

from dl4nlt import ROOT
from dl4nlt.datasets import denormalize_vec
# from dl4nlt.dataloader import norm_essay_set
from dl4nlt.datasets import load_dataset
from dl4nlt.models.lstm import CustomLSTM
from dl4nlt.models.lstm import *

from tensorboardX import SummaryWriter
from datetime import datetime

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score

##### DEFAULT PARAMS #####

OUTPUT_DIR = os.path.join(ROOT, "models/lstm/saved_data")
DATASET_DIR = os.path.join(ROOT, "data")

EXP_NAME = 'Glove300D'
DROPOUT = 0.5
RNN_TYPE = 'BLSTM'
BATCHSIZE = 128
EPOCHS = 20
LR = 0.0005
VALIDATION_BATCHSIZE = 500


def create_collate(use_elmo=False):
    def collate(batch):
        batch = sorted(batch, key=lambda b: -1 * len(b.tokenized))
        lengths = list(map(lambda b: len(b.tokenized), batch))  # Needed for pack_padded in lstm

        X = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(b.tokenized).reshape(-1) for b in batch])

        s = torch.LongTensor([b.essay_set for b in batch]).reshape(-1)

        y = torch.tensor([b.y for b in batch]).reshape(-1)

        return X, s, y, lengths

    if use_elmo:
        raise NotImplementedError

    return collate


def train(name, dataset, epochs, lr, batchsize, **kwargs):

    def run_epoch(data, epoch, is_eval=False):
        if is_eval:
            model.eval()
            data_len = valid_len
        else:
            model.train()
            data_len = train_len

        loss = 0
        pearson = 0
        spearman = 0
        kappa = 0
        aloss = 0
        apearson = 0
        aspearman = 0

        for i_batch, batch in enumerate(data):
            x, s, t, l = batch
            x = x.to(device)
            t = t.to(device)

            model.zero_grad()

            y = model(x, l)
            this_loss = torch.sqrt(criterion(y, t))

            if not is_eval:
                this_loss.backward()
                optimizer.step()
                writer.add_scalar('Iteration training loss', this_loss.item(), epoch * len(data) + i_batch)

            y_denorm = denormalize_vec(s, y.detach(), device)
            t_denorm = denormalize_vec(s, t.detach(), device)
            this_kappa = cohen_kappa_score(t_denorm, y_denorm, labels=list(range(1, 30)), weights="quadratic")
            this_pearson, p_value = pearsonr(t.detach(), y.detach())
            this_spearman, p_value = spearmanr(t.detach(), y.detach())

            if not is_eval:
                writer.add_scalar('Iteration training pearson', this_pearson, epoch * len(data) + i_batch)
                writer.add_scalar('Iteration training spearman', this_spearman, epoch * len(data) + i_batch)
                writer.add_scalar('Iteration training kappa', this_kappa, epoch * len(data) + i_batch)

            # Weight scores depending on batch size (last batch is smaller)
            loss += this_loss.item() * x.shape[1]
            pearson += this_pearson * x.shape[1]
            spearman += this_spearman * x.shape[1]
            kappa += this_kappa * x.shape[1]

            aloss += x.shape[1] * torch.sqrt(
                criterion(y_denorm.type(torch.FloatTensor).to(device), t_denorm.type(torch.FloatTensor).to(device)))
            apearson = x.shape[1] * pearsonr(t_denorm.type(torch.FloatTensor).to(device),
                                             y_denorm.type(torch.FloatTensor).to(device))[0]
            aspearman = x.shape[1] * spearmanr(t_denorm.type(torch.FloatTensor).to(device),
                                               y_denorm.type(torch.FloatTensor).to(device))[0]

        # Average over the size of the train/valid data
        loss /= data_len
        pearson /= data_len
        spearman /= data_len
        kappa /= data_len
        aloss /= data_len
        apearson /= data_len
        aspearman /= data_len

        return loss, pearson, spearman, kappa, aloss, apearson, aspearman

    ##############################################
    ## Data, Model and Optimizer initialization ##
    ##############################################

    outdir = os.path.join(OUTPUT_DIR, dataset, name)

    print('Outdir:', outdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outfile_metrics = os.path.join(outdir, "metrics.pickle")
    outfile_metrics_valid = os.path.join(outdir, "metrics_valid.csv")
    outfile_metrics_train = os.path.join(outdir, "metrics_train.csv")

    os.makedirs(outdir, exist_ok=True)

    writer = SummaryWriter(os.path.join('runs', dataset, name))

    training_set, validation_set, _ = load_dataset(os.path.join(DATASET_DIR, dataset))
    train_len, valid_len = len(training_set), len(validation_set)
    vocab_len = len(training_set.dict)
    training = DataLoader(training_set, batch_size=batchsize, shuffle=True, pin_memory=True,
                          collate_fn=create_collate())
    validation = DataLoader(validation_set, batch_size=VALIDATION_BATCHSIZE, shuffle=False, pin_memory=True,
                            collate_fn=create_collate())

    model = CustomLSTM(vocab_len=vocab_len, device=device, target_vocab_to_idx=training_set.dict.word2idx,  **kwargs)
    model.to(device)

    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr)
    # optimizer = SGD(model.parameters(), lr)

    ####################################
    ############# TRAINING #############
    ####################################
    start_time = time.time()

    print('\n###############################################')
    print('Starting epoch 0 (Random Guessing)')
    loss, pearson, spearman, kappa, aloss, apearson, aspearman = run_epoch(validation, 0, is_eval=True)
    print('| Valid Loss: {:.5f} |  Pearson: {:.5f} |  Spearman: {:.5f} |  Kappa: {:.5f} |\n'.format(loss, pearson, spearman, kappa))
    print('| Denor Loss: {:.5f} |  Pearson: {:.5f} |  Spearman: {:.5f} |\n'.format(aloss, apearson, aspearman))

    metrics = {
        "train": {
            "rmse": [],
            "pearson": [],
            "spearman": [],
            "kappa": [],
        },
        "valid": {
            "rmse": [],
            "pearson": [],
            "spearman": [],
            "kappa": [],
        }
    }

    for e in range(epochs):
        print('###############################################')
        print('Starting epoch {}'.format(e + 1))

        loss, pearson, spearman, kappa, aloss, apearson, aspearman = run_epoch(training, e, is_eval=False)
        print('| Train Loss: {:.5f} |  Pearson: {:.5f} |  Spearman: {:.5f} |  Kappa: {:.5f} |'.format(
            loss, pearson, spearman, kappa))
        print('| Denor Loss: {:.5f} |  Pearson: {:.5f} |  Spearman: {:.5f} |\n'.format(aloss, apearson, aspearman))
        metrics["train"] = update_metrics(metrics["train"], loss, pearson, spearman, kappa)
        update_writer(writer, e, loss, pearson, spearman, kappa, is_eval=False)
        update_csv(outfile_metrics_train, e, loss, pearson, spearman, kappa)

        loss, pearson, spearman, kappa, aloss, apearson, aspearman = run_epoch(validation, e, is_eval=True)
        print('| Valid Loss: {:.5f} |  Pearson: {:.5f} |  Spearman: {:.5f} |  Kappa: {:.5f} |'.format(
            loss, pearson, spearman, kappa))
        print('| Denor Loss: {:.5f} |  Pearson: {:.5f} |  Spearman: {:.5f} |\n'.format(aloss, apearson, aspearman))
        metrics["valid"] = update_metrics(metrics["valid"], loss, pearson, spearman, kappa)
        update_writer(writer, e, loss, pearson, spearman, kappa, is_eval=True)
        update_csv(outfile_metrics_valid, e, loss, pearson, spearman, kappa)

        update_saved_model(metrics, model, optimizer, e, outdir)
        update_metrics_pickle(metrics, outfile_metrics)

        print()

    print("Finished training in {:.1f} minutes ".format((time.time() - start_time) / 60))

    return min(metrics["valid"]["rmse"]), max(metrics["valid"]["kappa"])


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=EXP_NAME,
                        help='Name of the experiment (used for output file name)')
    parser.add_argument('--dataset', type=str, default='local_mispelled',
                        help='Name of the dataset to use')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=LR,
                        help='Value of learning rate')
    parser.add_argument('--batchsize', type=int, default=BATCHSIZE,
                        help='The batch-size for the training')

    parser.add_argument('--emb_size', type=int, default=300,
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
    parser.add_argument('--use_glove', dest='use_glove', action='store_true')
    parser.set_defaults(use_glove=False)

    FLAGS = parser.parse_args()
    params = vars(FLAGS)

    if 'emb_size' in params:
        params['embeddings'] = params['emb_size']
        del params['emb_size']

    if 'embeddings_path' in params:
        if params['embeddings_path'] is not None:
            params['embeddings'] = params['embeddings_path']
        del params['embeddings_path']

    train(**params)
