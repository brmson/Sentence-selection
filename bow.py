from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 08 18:34:49 2015

@author: Silvicek
"""
from vecfromtext import getGloveDict
import numpy as np


VLEN=50 #glove vector length


def prepForGrad(q,a1,a0,ans1,ans0,glovepath2):    
    """ Bag-of-words based embedding vectors for each sentence.
    Returns a matrix with one sentence per row. """
    gloveDict=getGloveDict(glovepath2)
    qa=np.zeros((len(q),50))
    a1a=np.zeros((len(a1),50))
    a0a=np.zeros((len(a0),50))
    for i in range(0,len(q)):
        qa[i][:]=boxSentence(q[i],gloveDict)
    print 'questions embedded'
    for i in range(0,len(a1)):
        a1a[i][:]=boxSentence(a1[i],gloveDict)
    print 'true answers embedded'
    for i in range(0,len(a0)):
        a0a[i][:]=boxSentence(a0[i],gloveDict)
    print 'false answers embedded'
    return (qa,a1a,a0a)

#Boxofwords for sentence
def boxSentence(sentence,gloveDict):
    i=0
    v=np.zeros(VLEN)
    for word in sentence:
        x=gloveDict.get(word)
        if x is not None:
            v+=x
            i+=1
    if i!=0:
        v=v/i
    return v
