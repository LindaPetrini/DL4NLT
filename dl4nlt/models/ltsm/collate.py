

import torch


def collate(batch):
    
    batch = sorted(batch, key=lambda b: -1*len(b.tokenized))
    
    return torch.nn.utils.rnn.pad_sequence([torch.LongTensor(b.tokenized).reshape(-1, 1) for b in batch])
