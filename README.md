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


Property selection in yodaqa/moviesC:
-------------------------------------

Folow these steps if you want to retrain currently used weights:
* run yodaqa (for example train-and-eval.sh) with system property cz.brmlab.yodaqa.dump\_property\_labels set as path to folder where you want to generate the data
* run './std_run.sh -p \[path\]' (path is the directory of dumped yodaqa files). You can alter the training constants in basicgrad.py and train.py
* if you are happy with the results, you can copy and rename generated file Mbtemp.txt located in data folder to yodaqa/src/main/resources/cz/brmlab/yodaqa/analysis/rdf/Mbprop.txt

**Used dataset:**  
train questions: 270 train sentences: 19624	(generated with curated-measure.sh train)  
test questions: 222 test sentences: 17561	(generated with curated-measure.sh train)  
2.7902739024% of the properties contains correct answers  
random test mrr = 0.0475542678953  

**Current results:**  
MMR after unigram learning train: 0.600856454434  
MMR after unigram learning test: 0.582881935037  

Sentence selection on yodaqa/curated:
-------------------------------------

Folow these steps if you want to retrain currently used weights:
* run yodaqa (for example train-and-eval.sh) with system property cz.brmlab.yodaqa.jacana set as path to folder where you want to generate the data
* run './std_run.sh \[path\]' (path is the directory of dumped yodaqa files). You can alter the training constants in basicgrad.py and train.py
* if you are happy with the results, you can copy and rename generated file Mbtemp.txt located in data folder to yodaqa/src/main/resources/cz/brmlab/yodaqa/analysis/passextract/Mb.txt

**Used dataset:**  
train questions: 186 train sentences: 43843	(generated with curated-measure.sh train)  
test questions: 429 test sentences: 88779	(generated with curated-measure.sh test)  
5.21294450264% of the properties contains correct answers  
random test mrr = 0.0760195275186  

**Current results:**  
baseline (clue1+0.25*clue2):  
MRR unigram+clues train 0.249327071552  
MRR unigram+clues test 0.29659580682  

glove only:  
MMR after unigram learning train: 0.224787152966  
MMR after unigram learning test: 0.222749753007  

glove+clue1:  
MRR unigram+clues train 0.358206351223  
MRR unigram+clues test 0.388948882077  

