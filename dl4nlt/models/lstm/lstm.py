import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

from dl4nlt.models.embeddings.glove_embedding import create_glove_embedding

def load_latest_sswe_embeddings(path):
    """
    loads the last sswe embedding layer saved
    embeddings = load_latest_sswe_embeddings()
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    vocab_len = checkpoint['state_dict']['embeddings.weight'].size()[0]
    emb_size = checkpoint['state_dict']['embeddings.weight'].size()[1]
    embedding_layer = torch.nn.Embedding(vocab_len, emb_size, padding_idx=0).to(device)
    embedding_layer.weight.data = checkpoint['state_dict']['embeddings.weight']
    return embedding_layer


class CustomLSTM(nn.Module):
    def __init__(self, vocab_len, embeddings, n_hidden_units, device, n_hidden_layers=1, n_output=1, dropout=0.5, rnn_type='LSTM', target_vocab_to_idx=None, use_glove=False):
        super(CustomLSTM, self).__init__()

        torch.manual_seed(42)

        self.device = device
        self.rnn_type = rnn_type
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.linear_size = 30
        self.output_size = n_output
        self.blstm = False

        self.drop = nn.Dropout(dropout)

        if isinstance(embeddings, str):
            self.init_emb_from_file(embeddings)
        elif target_vocab_to_idx is not None and use_glove is True:
            self.encoder = create_glove_embedding(target_vocab_to_idx=target_vocab_to_idx, emb_size=embeddings)
        elif embeddings is not None and isinstance(embeddings, int) and embeddings > 0:
            self.encoder = nn.Embedding(vocab_len, embeddings, padding_idx=0)
        else:
            raise ValueError("ERROR! 'embeddings' parameter not valid: it should be either a positive integer or a path to the embeddings file")

        if rnn_type in ['LSTM', 'GRU', 'BLSTM']:
            is_bidirectional = True if rnn_type == 'BLSTM' else False
            if rnn_type is 'BLSTM':
                self.blstm = True
                rnn_type = 'LSTM'
            self.rnn = getattr(nn, rnn_type)(self.encoder.embedding_dim, n_hidden_units, n_hidden_layers,
                                             dropout=dropout,
                                             bidirectional=is_bidirectional,
                                             batch_first=False)  # Dropout added only for n_layers > 1
        else:
            raise ValueError("An invalid option for `--model` was supplied, options are ['LSTM', 'GRU', 'BLSTM']")

        self.decoder = nn.Linear(n_hidden_units, self.linear_size)
        self.relu = nn.ReLU()
        self.decoder2 = nn.Linear(self.linear_size, n_output)
        self.sigmoid = nn.Sigmoid()

        if self.blstm:
            self.single_decoder = nn.Linear(n_hidden_units * 2, n_output)
        else:
            self.single_decoder = nn.Linear(n_hidden_units, n_output)



        # self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        if self.rnn_type == 'LSTM':
            return (weight.new(self.n_hidden_layers, bsz, self.n_hidden_units).zero_(),
                    weight.new(self.n_hidden_layers, bsz, self.n_hidden_units).zero_())
        else:
            return (weight.new(self.n_hidden_layers, bsz, self.n_hidden_units).zero_(),)

    def init_emb_from_file(self, path='dl4nlt/models/sswe/saved_models/local_mispelled/latest.pth.tar'):
        # emb_mat = np.genfromtxt(path)
        # - infer the embedding directly from the file, without building the module beforehand
        # self.encoder.weight.data.copy_(torch.from_numpy(emb_mat))
        self.encoder = load_latest_sswe_embeddings(path)

    def forward(self, input, l):

        emb = self.encoder(input)

        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, l)
        output, _ = self.rnn(packed)
        unpacked, pack_length = torch.nn.utils.rnn.pad_packed_sequence(output)

        if self.blstm:
            idx = pack_length.view(1, -1, 1).expand(1, -1, 2 * self.n_hidden_units).to(dtype=torch.long).to(self.device) - 1
        else:
            idx = pack_length.view(1, -1, 1).expand(1, -1, self.n_hidden_units).to(dtype=torch.long).to(self.device) - 1

        output = torch.gather(unpacked, 0, idx).squeeze(0)

        output = self.drop(output)

        # output = self.decoder(output)
        # output = self.decoder2(self.relu(output))
        output = self.single_decoder(output)
        # print(emb.shape, unpacked.shape, output.shape)

        # output = self.sigmoid(output)

        return output.view(-1)


if __name__ == "__main__":
    rnn = CustomLSTM(5, 10, 20, 1)
    input = torch.randint(0, 5, (4, 2)).type(torch.LongTensor)
    h0 = torch.randn(1, 2, 20)
    c0 = torch.randn(1, 2, 20)
    output, hn = rnn(input, (h0, c0))
    print(output)
