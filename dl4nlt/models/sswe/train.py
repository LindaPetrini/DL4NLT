import argparse
import os.path
import pickle

import torch

from torch.utils.data import DataLoader

from torch.nn import MSELoss

from torch.optim import Adam
from torch.optim import SGD

from dl4nlt import ROOT
OUTPUT_DIR = os.path.join(ROOT, "models/sswe/saved_models")
DATASET_DIR = os.path.join(ROOT, "data/")

from dl4nlt.datasets import load_dataset
from dl4nlt.models.sswe import SSWEModel

from dl4nlt.models.sswe.context_dataset import ContextDateset

ERROR_RATE = 200
CONTEXT_SIZE = 9
ALPHA = 0.1

EMBEDDINGS = 200
HIDDEN_UNITS = 100

BATCHSIZE = 1000
EPOCHS = 5
LR = 1e-4
        

def save_model_checkpoint(state, filename):
    """
    saves a model to filename - in folder called saved_models
    """
    torch.save(state, filename)


def main(dataset, epochs, lr, batchsize, context_size, error_rate, alpha, **kwargs):
    
    assert 0. <= alpha <= 1.
    assert error_rate > 0 and isinstance(error_rate, int)
    assert context_size > 1 and isinstance(context_size, int)
    
    outdir = os.path.join(OUTPUT_DIR, dataset)
    
    os.makedirs(outdir, exist_ok=True)

    outfile = os.path.join(outdir, 'latest.pth.tar')
    
    training, _, _ = load_dataset(os.path.join(DATASET_DIR, dataset))
    
    training = ContextDateset(training, context_size)
    
    vocab_len = len(training.dict)
    
    model = SSWEModel(vocab_len=vocab_len, context_size=context_size, **kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    def collate(batch):
        
        X = torch.stack([torch.LongTensor(b.ngram).reshape(-1) for b in batch])
        y = torch.tensor([b.y for b in batch]).reshape(-1)
        
        # add a dimension between the batch dim and the sequence dim for the noisy versions of each sequence (i.e. with
        # the word in the middle changed to a random word
    
        X = X.unsqueeze(1).expand(-1, error_rate + 1, -1).contiguous()
        
        # find the position of the word in the middle of the sequence
        t = context_size // 2
        
        # change the word in the middle of the sequences to a random word, for each batch and each copy of the original
        # sequence (notice that the first copy is left unchanged)
        
        X[:, 1:, t] = torch.randint(0, vocab_len-1, size=(X.shape[0], error_rate))
        
        return X, y
    
    training = DataLoader(training, batch_size=batchsize, shuffle=True, pin_memory=True, collate_fn=collate)

    score_loss = MSELoss()
    
    optimizer = Adam(model.parameters(), lr)
    # optimizer = SGD(model.parameters(), lr)
    
    model.to(device)

    train_losses = []

    for e in range(epochs):

        train_loss = 0

        print('###############################################')
        print('Starting epoch {}'.format(e + 1))
        
        model.train()
    
        for i_batch, batch in enumerate(training):
            if i_batch % 100 == 0:
                    print("{}/{}".format(i_batch, len(training)))
            
            x, t = batch            
            x = x.to(device)
            t = t.to(device)
        
            model.zero_grad()
        
            fc, fs = model(x)
                        
            # the loss for the score prediction applies only to the original sequence (i.e. the first one)
            score_l = score_loss(fs[:, 0], t)
            
            # the loss for the context
            score_c = torch.clamp(1 - fc[:, 0].view(-1, 1) + fc[:, 1:], min=0).mean()
            
            # the final loss is the weighted sum of these 2 losses
            l = (1 - alpha) * score_l + alpha * score_c

            l.backward()
            optimizer.step()
        
            train_loss += l.item() * x.shape[1]
            
            if i_batch % 10 == 0:
                print('| Training loss at {}: {} |'.format(i_batch, (train_loss/(i_batch+1))))
                
            
        train_loss /= len(training)
        train_losses.append(train_loss)
    
        print('| Training loss: {} |'.format(train_loss))
        
        print("Saving Model checkpoint")
        save_model_checkpoint({                
                'state_dict': model.state_dict(),              
            } , outfile)
            
        print('|\tModel Saved!')

        # print('Finished epoch {}'.format(e))
        # print('###############################################')
        # print('\n')


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, default='exp.model',
    #                     help='Name of the experiment (used for output file name)')
    parser.add_argument('--dataset', type=str, default='global_mispelled',
                        help='Name of the dataset to use')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=LR,
                        help='Value of learning rate')
    parser.add_argument('--batchsize', type=int, default=BATCHSIZE, help='The batch-size for the training')
    
    parser.add_argument('--context_size', type=int, default=CONTEXT_SIZE, help='The size of the context window for training embeddings')
    parser.add_argument('--error_rate', type=int, default=ERROR_RATE, help='The number of negative samples')
    parser.add_argument('--alpha', type=int, default=ALPHA, help='Weight parameter for the loss')
    
    parser.add_argument('--emb_size', type=int, default=EMBEDDINGS,
                        help='Size of the embeddings')
    parser.add_argument('--n_hidden_units', type=int, default=HIDDEN_UNITS,
                        help='Number of units per hidden layer')
    
    
    FLAGS = parser.parse_args()
    main(**vars(FLAGS))

