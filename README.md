# Predicting Essay Scores in the Kaggle ASAP Dataset using Deep Learning
#### Deep Learning for Natural Language Processing
#### University of Amsterdam, September-October 2018
#### Davide Belli, Gabriele Cesa, Alexander Geenen, Linda Petrini and Gautier Dagan

#### [https://github.com/LindaPetrini/DL4NLT](https://github.com/LindaPetrini/DL4NLT)

This is the repo for our Deep Learning for Natural Language Processing class Project, to run please unzip the data.zip folder and build the datasets using the following instructions:

1. Clone repo: `git clone git@github.com:LindaPetrini/DL4NLT.git`

1.1 Install requirements using conda or pip. Main libraries needed (if an external library is missing when training it must be because it is missing from this list):

  - torch 4.1
  - symspellpy
  - pandas
  - allennlp
  - wget
  - bcolz
  - tensorboardX
  - sklearn
  - gensim
  
  Note that you will need a version of python 3.6+ 
  
2. Generate the preprocessed datasets out of the tsv please run: 
  `cd DN4NLT`
  `python -m dl4nt.datasets.get_data`
  
  This step takes approximately 2 minutes on a modern computer, and is almost instant using ELMo since no preprocessing is done.

3. Run Baseline SVM model using doc2Vec on both the preprocessing techniques (refer to paper for more info): 
  - `python -m dl4nlt.models.w2v_baseline.build_doc2vec --dataset='global_mispelled'`
  - `python -m dl4nlt.models.w2v_baseline.build_doc2vec --dataset='local_mispelled'`
  - `python -m dl4nlt.models.w2v_baseline.svm_doc2vec --dataset='global_mispelled'`
  - `python -m dl4nlt.models.w2v_baseline.svm_doc2vec --dataset='local_mispelled'`
  
4. You can then train models using our run experiments files or alternatively train.py files directly. To train the LSTM/BLSTM/GRU models using fresh embedding layer of size 200 you can run:

  - `python -m dl4nlt.models.lstm.train --dataset='global_mispelled' --emb_size=200 --rnn_type='LSTM'`
  - `python -m dl4nlt.models.lstm.train --dataset='global_mispelled' --emb_size=200 --rnn_type='BLSTM'`
  - `python -m dl4nlt.models.lstm.train --dataset='global_mispelled' --emb_size=200 --rnn_type='GRU'`

4.1. To specifically use the SSWE emeddings you first need to train them by running: 

  - `python -m dl4nlt.models.sswe.train`
  - `python -m dl4nlt.models.lstm.train --dataset='global_mispelled' --emb_size='sswe'`
  
Again you can pass in arguments such as --dataset to select whether to train on global or local. Please have a look at the bottom of the train.py file in the sswe folder to find more about the different flags. 
   
5. To train and then evaluate the ELMo-GRU model please run (this default requires a GPU with memory of at least 8Gb): 
  - `python -m dl4nlt.models.embeddings.train` 
  - If you want to use a cpu for this it is possible to use a smaller batch size but by passing the `--batch-size=1` flag, but it is not officially supported at this time.  

This step will download the ELMo embeddings directly if they are not in the proper folder. The download will take approximately 370mb of space.

We have shown that using the recently published ELMo embeddings and without any tuning we are able to obtain state of the art accuracy on the ASAP-AES dataset in order to predict essay scores on standardized English tests. 

Please note that some of these training methods, especially the ELMo, use a lot of space and so require a gpu with a certain capacity. The baseline SVM model however can be run quite easily and fast on cpu (see step 3). Other neural network models might also require GPU if convergence in training is wanted in a reasonable amount of time. 

