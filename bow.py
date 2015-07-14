from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 08 18:34:49 2015

@author: Silvicek
"""

import numpy as np
from vecfromtext import getGloveVector

VLEN=50 #glove vector length

#Boxofwords for sentence
def boxSentence(sentence,glovepath2):
    i=0
    v=np.zeros(VLEN)
    for word in sentence:
        x=getGloveVector(word,glovepath2)
        if x is not None:
            v+=x
            i+=1
    v=v/i
    return v

#Numeric bow vectors from sentences
def prepForGrad(q,a1,a0,ans1,ans0,glovepath2):
    qa=np.zeros((len(q),50))
    a1a=np.zeros((len(a1),50))
    a0a=np.zeros((len(a0),50))
    for i in range(0,len(q)):
        if i%100==0:
            print 'working...'
        qa[i][:]=boxSentence(q[i],glovepath2)
    
    for i in range(0,len(a1)):
        if i%100==0:
            print 'working...'
        a1a[i][:]=boxSentence(a1[i],glovepath2)
    
    for i in range(0,len(a0)):
        if i%100==0:
            print 'working...'
        a0a[i][:]=boxSentence(a0[i],glovepath2)
    
    
    return (qa,a1a,a0a)
