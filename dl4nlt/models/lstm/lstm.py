import torch
import torch.nn as nn
from torch.nn import init
import numpy as np


class CustomLSTM(nn.Module):
    def __init__(self, vocab_len, emb_size, n_hidden_units, device, n_hidden_layers=1, n_output=1, dropout=0.5, rnn_type='LSTM',
                 embeddings_path=None):
        super(CustomLSTM, self).__init__()
        
        torch.manual_seed(42)
        
        self.device = device
        self.emb_size = emb_size
        self.rnn_type = rnn_type
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.linear_size = 30
        self.output_size = n_output
        
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_len, emb_size, padding_idx=0)
        if rnn_type in ['LSTM', 'GRU', 'BLTSM']:
            is_bidirectional = True if rnn_type == 'BLSTM' else False
            if rnn_type is 'BLSTM':
                rnn_type = 'LSTM'
            self.rnn = getattr(nn, rnn_type)(emb_size, n_hidden_units, n_hidden_layers,
                                             dropout=dropout,
                                             bidirectional=is_bidirectional,
                                             batch_first=False)  # Dropout added only for n_layers > 1
        else:
            raise ValueError("An invalid option for `--model` was supplied, options are ['LSTM', 'GRU', 'BLSTM']")
        
        self.decoder = nn.Linear(n_hidden_units, self.linear_size)
        self.relu = nn.ReLU()
        self.decoder2 = nn.Linear(self.linear_size, n_output)
        self.sigmoid = nn.Sigmoid()
        
        self.single_decoder = nn.Linear(n_hidden_units, n_output)
        
        if embeddings_path is not None:
            self.init_emb_from_file(embeddings_path)
        
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

    def init_emb_from_file(self, path):
        emb_mat = np.genfromtxt(path)
        self.encoder.weight.data.copy_(torch.from_numpy(emb_mat))
    
    def forward(self, input, hidden, l):
        
        emb = self.encoder(input)
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, l)
        output, hidden = self.rnn(packed)
        unpacked, pack_length = torch.nn.utils.rnn.pad_packed_sequence(output)
        
        idx = pack_length.view(1, -1, 1).expand(1, -1, self.n_hidden_units).to(dtype=torch.long).to(self.device) - 1
        output = torch.gather(unpacked, 0, idx).squeeze(0)
        
        output = self.drop(output)
        
        # output = self.decoder(output)
        # output = self.decoder2(self.relu(output))
        output = self.single_decoder(output)
        # print(emb.shape, unpacked.shape, output.shape)
        
        # output = self.sigmoid(output)
        
        return output.view(-1), hidden


if __name__ == "__main__":
    rnn = CustomLSTM(5, 10, 20, 1)
    input = torch.randint(0, 5, (4, 2)).type(torch.LongTensor)
    h0 = torch.randn(1, 2, 20)
    c0 = torch.randn(1, 2, 20)
    output, hn = rnn(input, (h0, c0))
    print(output)
