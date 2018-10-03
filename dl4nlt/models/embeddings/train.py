import argparse
import os
import time

import tensorboardX

import torch
from torch.utils.data import DataLoader


import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score


from dl4nlt import ROOT
from dl4nlt.datasets.dataset import ASAP_Data, denormalize_vec
from dl4nlt.models.embeddings.embedding_gru import EmbeddingGRU


def create_collate(use_elmo=True):
    def elmo_collate(batch):
        sorted_batch = sorted(batch, key=lambda b: -1 * len(b.essay))

        essays = list(map(lambda b: b.essay, sorted_batch))

        essay_sets = torch.LongTensor([b.essay_set for b in batch]).reshape(-1)

        lengths = list(map(lambda b: len(b.essay), sorted_batch))

        targets = torch.LongTensor([b.y for b in sorted_batch]).reshape(-1)
        print([b.y_original for b in sorted_batch])

        return essays, lengths, essay_sets, targets

    return elmo_collate if use_elmo else lambda x: x


def train(config):
    assert config.embedding_type in ("ELMO", "GLOVE")

    cuda_enabled = False
    if 'cuda' in config.device:
        cuda_enabled = True

    print("Starting Training...")

    training_data = ASAP_Data(list(range(1, 9)), elmo_formatting=True)
    training_dataloader = DataLoader(training_data, batch_size=config.batch_size,
                                     shuffle=True, pin_memory=True, collate_fn=create_collate())

    print("Training data loaded...")

    if config.embedding_type is "ELMO":
        model = EmbeddingGRU(
            input_size=100,
            hidden_size=config.rnn_cell_dim,
            n_layers=config.num_rnn_layers,
            dropout=0.1,
            device=config.device,
            elmo=True,
        )
    else:
        model = None

    writer = tensorboardX.SummaryWriter(os.path.join(os.path.dirname(__file__), f"runs/ELMORUN{time.time()}"))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    if cuda_enabled:
        model.cuda(device=config.device)
        criterion.cuda(device=config.device)

    for e in range(config.num_epochs):
        print(f"Starting epoch {e}")

        loss, denorm_loss,\
        pearson, denorm_pearson,\
        spearman, denorm_spearman,\
            kappa = run_epoch(config, criterion, e, model, optimizer, training_data, training_dataloader, writer)
        print(f"Epoch loss {epoch_loss}")
        print(f"Epoch kappa {epoch_kappa}")


def run_epoch(config, criterion, e, model, optimizer, data, dataloader, writer, is_training=True):
    model.train() if is_training else model.eval()

    epoch_loss = 0
    epoch_denorm_loss = 0
    epoch_pearson = 0
    epoch_denorm_pearson = 0
    epoch_spearman = 0
    epoch_denorm_spearman = 0
    epoch_kappa = 0

    for batch_num, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch_input, lengths, essay_sets, targets = batch
        targets = targets.float().to(config.device)

        outputs, hidden = model(batch_input, lengths)

        loss = torch.sqrt(criterion(outputs, targets))

        if is_training:
            loss.backward()
            optimizer.step()
            writer.add_scalar('Iteration training loss', float(loss.item()), e * len(dataloader) + batch_num)

        y_denorm = denormalize_vec(essay_sets, outputs.detach(), config.device)
        t_denorm = denormalize_vec(essay_sets, targets.detach(), config.device)
        print(essay_sets)
        print(targets)
        print(y_denorm, t_denorm)
        kappa = cohen_kappa_score(t_denorm, y_denorm, labels=list(range(0, 30)), weights="quadratic")
        pearson, p_value = pearsonr(targets.detach(), outputs.detach())
        spearman, p_value = spearmanr(targets.detach(), outputs.detach())

        if is_training:
            writer.add_scalar('Iteration training pearson', pearson, e * len(data) + batch_num)
            writer.add_scalar('Iteration training spearman', spearman, e * len(data) + batch_num)
            writer.add_scalar('Iteration training kappa', kappa, e * len(data) + batch_num)

        print(f"Batch loss {float(loss.item())}")
        print(f"Cohen kappa {kappa}")
        batch_len = len(batch_input)
        epoch_loss += float(loss.item()) * batch_len
        epoch_denorm_loss += torch.sqrt(
            criterion(t_denorm.float().to(config.device), y_denorm.float().to(config.device))
        )
        epoch_pearson += pearson * batch_len
        epoch_denorm_pearson += batch_len * pearsonr(t_denorm.float().to(config.device),
                                         y_denorm.float().to(config.device))[0]
        epoch_spearman += spearman * batch_len
        epoch_denorm_spearman += batch_len * spearmanr(t_denorm.float().to(config.device),
                                           y_denorm.float().to(config.device))[0]
        epoch_kappa += kappa * batch_len

    data_length = len(data)
    epoch_loss /= data_length
    epoch_denorm_loss /= data_length
    epoch_pearson /= data_length
    epoch_denorm_pearson /= data_length
    epoch_spearman /= data_length
    epoch_denorm_spearman /= data_length
    epoch_kappa /= data_length

    return epoch_loss, epoch_denorm_loss, epoch_pearson, epoch_denorm_pearson, epoch_spearman, epoch_denorm_spearman, epoch_kappa


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    DATASET_DIR = os.path.join(ROOT, "data/baseline")

    default_device = "cpu"
    if torch.cuda.is_available():
        default_device = "cuda:0"

    parser.add_argument('--device', type=str, default=default_device,
                        help='Device to run computations on')
    parser.add_argument('--dataset', type=str, default=DATASET_DIR,
                        help='Path to the folder containg the dataset')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate for training')
    parser.add_argument('--embedding_type', type=str, default='ELMO',
                        help='Type of Embedding to use (ELMO, GLOVE)')
    parser.add_argument('--rnn_cell_dim', type=int, default=80,
                        help='Size of hidden dimension for the RNN cells')
    parser.add_argument('--num_rnn_layers', type=int, default=1,
                        help='Number of RNN layers')
    parser.add_argument('--rnn_dropout', type=float, default=0.1,
                        help='Dropout probability between RNN layers')

    config = parser.parse_args()
    train(config)
