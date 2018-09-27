import torch
import torch.nn as nn
from torch.nn import init
import numpy as np


class SSWEModel(nn.Module):
    def __init__(self,
                 vocab_len,
                 context_size,
                 emb_size,
                 n_hidden_units):
        super(SSWEModel, self).__init__()
        
        torch.manual_seed(42)
        
        self.emb_size = emb_size
        self.context_size = context_size
        self.n_hidden_units = n_hidden_units
        
        self.embeddings = nn.Embedding(vocab_len, emb_size, padding_idx=0)
        
        self.emb_to_hidden = nn.Linear(emb_size*context_size, n_hidden_units)
        
        self.hidden_to_context = nn.Linear(n_hidden_units, 1)
        self.hidden_to_score = nn.Linear(n_hidden_units, 1)
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        # self.emb_to_hidden.weight.data
        # self.emb_to_hidden.bias.data.fill_(0)
        # self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, input):
        
        shape = list(input.shape)
        shape[-1] = -1

        # print(input.shape, shape)
        
        s = self.embeddings(input).view(*shape)
        
        # print(s.shape)
        
        i = torch.nn.functional.hardtanh(self.emb_to_hidden(s))
        
        # print(i.shape)
        # context score predictions
        fc = self.hidden_to_context(i).squeeze()
        
        # essays' scores predictions
        fs = self.hidden_to_score(i).squeeze()
        
        # print(fc.shape, fs.shape)
        
        return fc, fs
