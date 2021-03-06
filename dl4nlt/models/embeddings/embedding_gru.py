import torch
import torch.nn as nn

from allennlp.modules.elmo import batch_to_ids

from dl4nlt.models.embeddings.elmo_embedding import create_elmo_embedding


class EmbeddingGRU(nn.Module):
    """
    ELMo powered GRU network. Uses a linear projection from the base ELMo embeddings to a different desired
    dimensionality. This projection is then used as input to the GRU network in order to make predictions.
    """
    def __init__(self, input_size, hidden_size, n_layers=2, n_outputs=1, dropout=0.1, device='cpu'):
        super(EmbeddingGRU, self).__init__()

        self.input_size = 1024
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        self.embedding = create_elmo_embedding()
        self.embedding_projection = nn.Linear(1024, input_size)

        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        self.linear = nn.Linear(hidden_size, n_outputs)

    def forward(self, input_seqs, input_lengths, hidden=None):
        input_seqs = batch_to_ids(input_seqs).to(self.device)
        
        embedded, _ = self.elmo_to_torch(self.embedding(input_seqs))
        embedded = embedded[0].to(self.device).transpose(0, 1)
        emb_projection = self.embedding_projection(embedded)

        packed = torch.nn.utils.rnn.pack_padded_sequence(emb_projection, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        idx = output_lengths.view(1, -1, 1).expand(1, -1, self.hidden_size).to(dtype=torch.long).to(self.device) - 1
        outputs = torch.gather(outputs, 0, idx).squeeze(0)

        outputs = self.linear(outputs)
        
        return outputs.squeeze(), hidden

    @staticmethod
    def elmo_to_torch(elmo_embedding):
        """
        Extract the elmo embedding tensor and the corresponding sequence lengths from a AllenNLP ELMo Module's output
        :param elmo_embedding: ELMo Module Output
        :return: Extracted tensor and sequence lengths
        """
        tensor = elmo_embedding["elmo_representations"]
        seq_lengths = torch.sum(elmo_embedding["mask"], 1)

        return tensor, seq_lengths

