import argparse
import os
import time

import tensorboardX

import torch
from torch.utils.data import DataLoader

from dl4nlt.datasets import load_dataset

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score


from dl4nlt import ROOT
from dl4nlt.datasets.dataset import denormalize_vec
from dl4nlt.models.embeddings.embedding_gru import EmbeddingGRU


VALIDATION_BATCHSIZE = 500


def elmo_collate(batch):
    sorted_batch = sorted(batch, key=lambda b: -1 * len(b.elmo_tokenized))
    
    essays = list(map(lambda b: b.elmo_tokenized, sorted_batch))
    
    essay_sets = torch.LongTensor([b.essay_set for b in sorted_batch]).reshape(-1)

    lengths = list(map(lambda b: len(b.elmo_tokenized), sorted_batch))

    targets = torch.FloatTensor([b.y for b in sorted_batch]).reshape(-1)

    return essays, lengths, essay_sets, targets


def train(config):

    cuda_enabled = False
    if 'cuda' in config.device:
        cuda_enabled = True

    print("Starting Training...")

    training_data, validation_data, _ = load_dataset(config.dataset)

    training_dataloader = DataLoader(training_data, batch_size=config.batch_size,
                                     shuffle=True, collate_fn=elmo_collate)

    # validation_dataloader = DataLoader(validation_data, batch_size=VALIDATION_BATCHSIZE, shuffle=False, pin_memory=True,
    #                         collate_fn=elmo_collate)
    
    print("Training data loaded...")

    model = EmbeddingGRU(
        input_size=100,
        hidden_size=config.rnn_cell_dim,
        n_layers=config.num_rnn_layers,
        dropout=0.1,
        device=config.device,
    )
    
    os.makedirs(os.path.join(os.path.dirname(__file__), "runs"), exist_ok=True)
    
    writer = tensorboardX.SummaryWriter(os.path.join(os.path.dirname(__file__), f"runs/ELMORUN{time.time()}"))
    
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    if cuda_enabled:
        model.cuda(device=config.device)
        criterion.cuda(device=config.device)

    for e in range(config.num_epochs):
        print(f"Starting epoch {e}")

        loss, denorm_loss, \
        pearson, denorm_pearson, \
        spearman, denorm_spearman, \
        kappa = run_epoch(config, criterion, e, model, optimizer, training_dataloader, writer)
        print(f"Epoch loss {epoch_loss}")
        print(f"Epoch kappa {epoch_kappa}")

        # val_loss, val_denorm_loss, \
        # val_pearson, val_denorm_pearson, \
        # val_spearman, val_denorm_spearman, \
        # val_kappa = run_epoch(config, criterion, e, model, optimizer, validation_dataloader, writer, is_training=False)


def run_epoch(config, criterion, e, model, optimizer, dataloader, writer, is_training=True):
    model.train() if is_training else model.eval()

    all_predictions = []
    all_targets = []
    
    all_predictions_denorm = []
    all_targets_denorm = []

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
        print(outputs.detach().tolist(), targets)
        print(y_denorm, t_denorm)
        kappa = cohen_kappa_score(t_denorm, y_denorm, labels=list(range(0, 30)), weights="quadratic")
        pearson, p_value = pearsonr(targets.detach(), outputs.detach())
        spearman, p_value = spearmanr(targets.detach(), outputs.detach())

        if is_training:
            writer.add_scalar('Iteration training pearson', pearson, e * len(dataloader) + batch_num)
            writer.add_scalar('Iteration training spearman', spearman, e * len(dataloader) + batch_num)
            writer.add_scalar('Iteration training kappa', kappa, e * len(dataloader) + batch_num)

        print(f"Batch loss {float(loss.item())}")
        print(f"Cohen kappa {kappa}")

        all_predictions_denorm += y_denorm.tolist()
        all_targets_denorm += t_denorm.tolist()

        all_predictions += outputs.tolist()
        all_targets += targets.tolist()
    
    epoch_loss = torch.sqrt(criterion(torch.FloatTensor(all_predictions), torch.FloatTensor(all_targets))).item()
    epoch_denorm_loss = torch.sqrt(criterion(torch.FloatTensor(all_predictions_denorm), torch.FloatTensor(all_targets_denorm))).item()
    
    epoch_pearson, _ = pearsonr(all_targets, all_predictions)
    epoch_denorm_pearson, _ = pearsonr(all_targets_denorm, all_predictions_denorm)
    
    epoch_spearman, _ = spearmanr(all_targets, all_predictions)
    epoch_denorm_spearman, _ = spearmanr(all_targets_denorm, all_predictions_denorm)

    epoch_kappa = cohen_kappa_score(all_targets_denorm, all_predictions_denorm, labels=list(range(0, 30)), weights="quadratic")

    return epoch_loss, epoch_denorm_loss, epoch_pearson, epoch_denorm_pearson, epoch_spearman, epoch_denorm_spearman, epoch_kappa


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    DATASET_DIR = os.path.join(ROOT, "data/elmo")

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
    parser.add_argument('--rnn_cell_dim', type=int, default=80,
                        help='Size of hidden dimension for the RNN cells')
    parser.add_argument('--num_rnn_layers', type=int, default=1,
                        help='Number of RNN layers')
    parser.add_argument('--rnn_dropout', type=float, default=0.1,
                        help='Dropout probability between RNN layers')

    config = parser.parse_args()
    train(config)
