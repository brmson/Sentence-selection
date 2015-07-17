Deep Learning for Answer Sentence Selection Reconstruction
==========================================================

This work started as an attempt to reproduce Yu et al.'s http://arxiv.org/abs/1412.1632

Used word embeddings: pre-trained GloVe vectors from http://nlp.stanford.edu/projects/glove/

Used dataset: TREC-based originally by Wang et al., 2007; in the form
by Yao et al., 2013 as downloaded from https://code.google.com/p/jacana/

So far implemented:
  * Bag of words + basic gradient descent learning classification
  * Bag of words + basic gradient descent learning classification + word counts logistic regression

Preprocessing (not required):
  * Run save.py first with updated filepath constants(const.py) if you have different dataset(requires jacana formating)

Train and test:
  * Run train.py for training from TREC TRAIN dataset and testing from TREC TEST dataset (generates truth.txt and res.txt, to evaluate run trec_eval with arguments -a truth.txt res.txt)

TODO:
  * CNN instead of bag of words unigram averaging for aggregate embeddings.
  * Train on Yao's TRAIN-ALL.
  * Re-evaluate using stock TREC scripts. 
