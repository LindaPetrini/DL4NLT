
from itertools import product
import pandas as pd

import os.path

from datetime import datetime

from dl4nlt import ROOT

from dl4nlt.models.lstm import train

OUTPUT_FILE = os.path.join(ROOT, "models/lstm/experiments/exp.csv")
DATASET_DIR = os.path.join(ROOT, "data/baseline")

# TODO set this path
EMB_FILE = os.path.join(ROOT, "models/sswe/saved_models", 'embedding')

params = {
    'rnn_type': ['LSTM', 'BLTSM'],
    'embeddings': [200, EMB_FILE],
    'lr': [1e-2, 1e-3, 1e-4],
    'n_hidden_units': [10, 64, 128],
    'dropout': [0.3, 0.5],
    'n_hidden_layers': [1], #[1, 2],
}

experiments_keys = list(params.keys())

defaults = {
    # 'name': 'exp.model',
    'dataset': DATASET_DIR,
    'epochs': 40,
    'lr': 3e-3,
    'batchsize': 128,
    'emb_size': 200,
    'n_hidden_units': 100,
    'n_hidden_layers': 1,
    'dropout': 0.1,
    'rnn_type': 'LSTM',
    'embeddings_path': None,
}

for k, v in defaults.items():
    if k not in params:
        params[k] = [v]

params = list(params.items())

params_names = [p[0] for p in params]
params_vals = [p[1] for p in params]


def save_results(logs):
    print('Saving...')
    logs = pd.DataFrame(data=logs, columns=params_names + ['Validation Loss', 'Validation Cohen', 'ModelPath'])
    
    logs = logs[sorted(logs.columns[:-3]) + logs.columns[-3:]]
    
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
    print(c, '\n')

    conf = {k: v for k, v in zip(params_names, c)}

    name = "(" + ";".join([str(conf[k]) for k in experiments_keys]) + ")"
    exp_name = 'runs/{}_{}'.format(name, datetime.now().strftime("%Y-%m-%d %H:%M"))

    conf['name'] = exp_name
    
    min_loss, max_cohen, outfile = train(conf)
    
    row = list(c) + [min_loss, max_cohen, outfile]
    logs.append(row)
    
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('\n')

    save_results(logs)



