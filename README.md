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
  * Run train.py for training from TREC TRAIN dataset and testing from TREC TEST dataset
  * train.py generates truth.txt and res.txt, to evaluate using the official trec_eval tool, run

	trec_eval -a truth.txt res.txt

TODO:
  * CNN instead of bag of words unigram averaging for aggregate embeddings. 

Results (evaluated using stock TREC scripts):

|                 | MRR    | MAP    |
|-----------------|--------|--------|
| TRAIN           | 0.7312 | 0.6551 |
| TRAIN-ALL       | 0.7308 | 0.6566 |
| TRAIN+count     | 0.7763 | 0.7165 |
| TRAIN-ALL+count | 0.8128 | 0.7258 |

==========================================================

 Now usable for sentence and/or property selection in yodaqa.
 
 For retraining on new data first use reparse.py or propreparse, then save.py then train.py.
 
 Don't forget to update path names.