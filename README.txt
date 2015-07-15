===Deep Learning for Answer Sentence Selection Reconstruction===
Original article: http://arxiv.org/abs/1412.1632
Used word embeddings: pre-trained GloVe vectors from http://nlp.stanford.edu/projects/glove/
Used TREC dataset downloaded from https://code.google.com/p/jacana/
================================================================

So far implemented:
*Box of words+basic gradient descent learning classification
*Box of words+basic gradient descent learning classification+word counts

Usage:  Run train.py for training from TREC TRAIN dataset and testing from TREC TEST dataset
        Run save.py first with updated filepath constants(const.py) if you have different dataset(requires jacana formating)