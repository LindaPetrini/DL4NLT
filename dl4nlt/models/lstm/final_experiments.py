
from itertools import product
import pandas as pd

import os.path

from datetime import datetime

from dl4nlt import ROOT

from dl4nlt.models.lstm.train import train

os.makedirs(os.path.join(ROOT, "models/lstm/experiments/"), exist_ok=True)

OUTPUT_FILE = os.path.join(ROOT, "models/lstm/experiments/the_final_experiments.csv")

params = {
    'rnn_type': ['LSTM', 'BLSTM', 'GRU'],
    'embeddings': ['sswe'],
    'dataset': ['local_mispelled'],
    'lr': [1e-4],
    'n_hidden_units': [128],
    'dropout': [0.4],
    'n_hidden_layers': [1],
    'epochs': [1],
}


experiments_keys = list(params.keys())

defaults = {
    # 'name': 'exp.model',
    'dataset': 'local_mispelled',
    'epochs': 100,
    'lr': 3e-3,
    'batchsize': 128,
    'n_hidden_units': 100,
    'n_hidden_layers': 1,
    'dropout': 0.1,
    'rnn_type': 'LSTM',
    'embeddings': 200,
}

for k, v in defaults.items():
    if k not in params:
        params[k] = [v]

params = list(params.items())

params_names = [p[0] for p in params]
params_vals = [p[1] for p in params]


def save_results(logs):
    print('Saving...')
    
    metrics = ['Validation Loss', 'Validation Cohen']\
              + ['Test Loss', 'Test Pearson', 'Test Spearman', 'Test Kappa', 'Test Denorm Loss', 'Test Denorm Pearson', 'Test Denorm Spearman']
    
    columns = params_names + metrics
    
    logs = pd.DataFrame(data=logs, columns=columns)
    
    logs = logs[sorted(params_names) + metrics]
    
    with open(OUTPUT_FILE, 'w') as f:
        logs.to_csv(f, sep=',', encoding='utf-8')
    
    print('...Saved!')


tot = 1

for p in params_vals:
    tot *= len(p)

combinations = product(*params_vals)

logs = []
for i, c in enumerate(combinations):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('{}/{}'.format(i, tot))
    # print(c, '\n')

    conf = {k: v for k, v in zip(params_names, c)}
    
        
    name = "(" + ";".join([str(conf[k]) for k in experiments_keys]) + ")"
    
    exp_name = name

    conf['name'] = exp_name
    print(conf, '\n')
    min_loss, max_cohen, test_metrics = train(**conf)
    
    row = list(c) + [min_loss, max_cohen] + list(test_metrics)
    logs.append(row)
    
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('\n')

    save_results(logs)



