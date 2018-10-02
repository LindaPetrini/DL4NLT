import argparse
import os
import time

import tensorboardX

import torch
from torch.utils.data import DataLoader

from dl4nlt import ROOT
from dl4nlt.datasets.dataset import ASAP_Data
from dl4nlt.models.embeddings.embedding_gru import EmbeddingGRU


def create_collate(use_elmo=True):
    def elmo_collate(batch):
        sorted_batch = sorted(batch, key=lambda b: -1 * len(b.essay))

        essays = list(map(lambda b: b.essay, sorted_batch))

        lengths = list(map(lambda b: len(b.essay), sorted_batch))

        targets = torch.LongTensor([b.y for b in sorted_batch]).reshape(-1)

        return essays, lengths, targets

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

    writer = tensorboardX.SummaryWriter(f"runs/ELMORUN{time.time()}")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    if cuda_enabled:
        model.cuda(device=config.device)
        criterion.cuda(device=config.device)

    for e in range(config.num_epochs):
        print(f"Starting epoch {e}")

        epoch_loss = 0

        for batch_num, batch in enumerate(training_dataloader):
            optimizer.zero_grad()

            batch_input, lengths, targets = batch
            targets = targets.float().to(config.device)

            outputs, hidden = model(batch_input, lengths)

            loss = torch.sqrt(criterion(outputs, targets))

            loss.backward()
            optimizer.step()
            writer.add_scalar('Iteration training loss', float(loss.item()), e * len(training_dataloader) + batch_num)

            print(f"Batch loss {float(loss.item())}")
            epoch_loss += float(loss.item()) * len(batch_input)

        epoch_loss /= len(training_data)
        print(f"Epoch loss {epoch_loss}")


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
    parser.add_argument('--batch_size', type=int, default=8,
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
