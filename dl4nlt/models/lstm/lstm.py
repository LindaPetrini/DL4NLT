import torch
import torch.nn as nn
from torch.nn import init
import numpy as np


class CustomLSTM(nn.Module):
    def __init__(self, vocab_len, emb_size, n_hidden_units, n_hidden_layers=1, n_output=1, dropout=0.2, rnn_type='LSTM',
                 embeddings_path=None):
        super(CustomLSTM, self).__init__()
        
        torch.manual_seed(42)
        
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
            self.rnn = getattr(nn, rnn_type)(emb_size, n_hidden_units, n_hidden_layers,
                                             dropout=dropout,
                                             bidirectional=is_bidirectional)  # Dropout added only for n_layers > 1
        else:
            raise ValueError("An invalid option for `--model` was supplied, options are ['LSTM', 'GRU', 'BLSTM']")
        
        self.decoder = nn.Linear(n_hidden_units, self.linear_size)
        self.decoder2 = nn.Linear(self.linear_size, n_output)
        self.softmax = nn.Sigmoid()
        
        if embeddings_path is not None:
            self.init_emb_from_file(embeddings_path)
        
        # self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output[-1, :, :])  # Take last prediction from the sequence
        
        output = self.decoder(output.view(-1, self.n_hidden_units))
        output = self.decoder2(output)
        
        # output = self.softmax(output)
        result = output.view(-1)
        return result, hidden
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        
        if self.rnn_type == 'LSTM':
            return (weight.new(self.n_hidden_layers, bsz, self.n_hidden_units).zero_(),
                    weight.new(self.n_hidden_layers, bsz, self.n_hidden_units).zero_())
        else:
            return weight.new(self.n_hidden_layers, bsz, self.n_hidden_units).zero_()
    
    def init_emb_from_file(self, path):
        emb_mat = np.genfromtxt(path)
        self.encoder.weight.data.copy_(torch.from_numpy(emb_mat))


if __name__ == "__main__":
    rnn = CustomLSTM(5, 10, 20, 1)
    input = torch.randint(0, 5, (4, 2)).type(torch.LongTensor)
    h0 = torch.randn(1, 2, 20)
    c0 = torch.randn(1, 2, 20)
    output, hn = rnn(input, (h0, c0))
    print(output)
