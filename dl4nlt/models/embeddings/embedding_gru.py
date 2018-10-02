import torch
import torch.nn as nn

from allennlp.modules.elmo import batch_to_ids

from dl4nlt.models.embeddings.elmo_embedding import create_elmo_embedding
from dl4nlt.models.embeddings.glove_embedding import create_glove_embedding


class EmbeddingGRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, n_outputs=1, dropout=0.1, elmo=True, target_vocab=None, device='cpu'):
        super(EmbeddingGRU, self).__init__()

        self.use_elmo = elmo
        self.input_size = input_size
        if self.use_elmo:
            self.input_size = 1024
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        if elmo:
            self.embedding = create_elmo_embedding()
        else:
            self.embedding = create_glove_embedding(target_vocab=target_vocab)

        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        self.linear = nn.Linear(hidden_size, n_outputs)

    def forward(self, input_seqs, input_lengths, hidden=None):
        input_seqs = batch_to_ids(input_seqs)
        embedded, _ = self.elmo_to_torch(self.embedding(input_seqs))
        embedded = embedded[0].to(self.device).transpose(0, 1)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        idx = output_lengths.view(1, -1, 1).expand(1, -1, self.hidden_size).to(dtype=torch.long).to(self.device) - 1
        outputs = torch.gather(outputs, 0, idx).squeeze(0)

        outputs = self.linear(outputs)
        return outputs.squeeze(), hidden

    @staticmethod
    def elmo_to_torch(elmo_embedding):
        tensor = elmo_embedding["elmo_representations"]
        seq_lengths = torch.sum(elmo_embedding["mask"], 1)

        return tensor, seq_lengths

