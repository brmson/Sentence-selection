from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 08 18:34:49 2015

@author: Silvicek
"""

import numpy as np
from vecfromtext import getGloveVector
import scipy.spatial.distance as sp

VLEN=50

#Boxofwords for sentence
def boxSentence(sentence):
    i=0
    v=np.zeros(VLEN)
    for word in sentence:
        x=getGloveVector(word)
        if x is not None:
            v+=x
            i+=1
    v=v/i
    return v

#MRR of cos-distance of bow
def bowcosmrr(q,a0,a1):
    x=[]
    ones=len(a1)
    zeros=len(a0)
    for sentence in a1:
        if len(sentence)>0:
            x.append(sp.cosine(boxSentence(sentence),boxSentence(q)))
    for sentence in a0:
        if len(sentence)>0:
            x.append(sp.cosine(boxSentence(sentence),boxSentence(q)))
        
    for i in range(0,zeros):
        if x.index(max(x))<ones:
            print 'found the truth sentence in iteration',i
            return 1/(i+1)
        x.remove(max(x))
    return 1/(zeros+1)

def bcmrrAll(q,a1,a0,ans1,ans0):
    x=0
    ones=0
    zeros=0
    for i in range(0,len(ans0)):
        x+=bowcosmrr(q[i],a0[zeros:zeros+ans0[i]],a1[ones:ones+ans1[i]])
        print x
        ones+=ans1[i]
        zeros+=ans0[i]
    return x/len(ans0)
    
    
#Numeric bow vectors from sentences
def prepForGrad(q,a1,a0,ans1,ans0):
    qa=np.zeros((len(q),50))
    a1a=np.zeros((len(a1),50))
    a0a=np.zeros((len(a0),50))
    for i in range(0,len(q)):
        if i%100==0:
            print 'working...'
        qa[i][:]=boxSentence(q[i])
    
    for i in range(0,len(a1)):
        if i%100==0:
            print 'working...'
        a1a[i][:]=boxSentence(a1[i])
    
    for i in range(0,len(a0)):
        if i%100==0:
            print 'working...'
        a0a[i][:]=boxSentence(a0[i])
    
    
    return (qa,a1a,a0a)
