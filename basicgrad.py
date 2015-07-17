from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 09:49:42 2015

@author: Silvicek
"""
import numpy as np
import matplotlib as mpl
import scipy.special as s

l=5e-3    #regularisation constant
alpha=2e-5  #learning constant

#Holds question with all its answers and T/F values as well as counted probabilities
class q(object):
    """Holds question with all its answers and T/F values as well as counted probabilities"""
    q=[]
    a=[]
    y=[]
    t=[]
    tcount=[]
    qtext=[]
    atext=[]
    counts=[]
    idf=[]
    def __init__(self,q,a1,a0,qtext,atext1,atext0):
        self.q=np.transpose(np.array(q,ndmin=2))  # question emb. (column)
        a1=np.array(a1,ndmin=2)  # correct ans. emb. (answers in rows)
        a0=np.array(a0,ndmin=2)  # incorrect
        self.a=np.hstack((np.transpose(a1),np.transpose(a0)))  # answer matrix (answer per column, correct come first)
        self.y=np.hstack((np.ones(len(a1)),np.zeros(len(a0))))  # answer labels
        self.qtext=qtext
        self.atext=atext1
        self.atext.extend((atext0))
        self.setCounts()
    def sett(self,M,b):
        """ compute answer labels based on model M,b  """
        self.t=s.expit(z(self.q,M,self.a,b)[0])  # answer labels as estimated by the model
    def settcount(self,results):
        self.tcount=results
    def setCounts(self):
        """ compute counts of common words in question and each answer """
        N=len(self.y)
        self.counts=np.zeros(len(self.y))
        self.idf=np.zeros(len(self.y))
        for i in range(0,len(self.counts)):
            for word in self.qtext:
                wc=self.atext[i].tolist().count(word)
                self.counts[i]+=wc
                if wc>0:
                    d=0
                    for sentence in self.atext:
                        if word in sentence:
                            d+=1
                    self.idf[i]+=wc*np.log(N/d)

def ttlist(qa,a1a,a0a,ans1,ans0,sentences):
    """Returns list of qs"""
    (questions,answers1,answers0)=sentences
    li=[]
    ones=0
    zeros=0
    for i in range(0,len(ans1)):
        li.append(q(qa[i],a1a[ones:ones+ans1[i]],a0a[zeros:zeros+ans0[i]],questions[i],
                    answers1[ones:ones+ans1[i]],answers0[zeros:zeros+ans0[i]]))
        ones+=ans1[i]
        zeros+=ans0[i]
    return li
    
def testGrad(M,b,li):
    """Updates weights using basic gradient descent"""
    bestmrr=0.0
    bestM=0
    bestb=0
    plot=np.zeros(100)
    for i in range (0,500):
        ggM=0
        ggb=0
        if i%5==0:
            plot[i/5]=lossAll(li,M,b)
            print 'learning.. loss function:',plot[i/5]
        for q in li:
            for j in range(0,len(q.y)):
                (gM,gb)=grad(q.y[j],q.q,M,np.transpose(np.array(q.a[:,j],ndmin=2)),b)
                ggM+=gM
                ggb+=gb
        M=M-alpha*ggM
        b=b-alpha*ggb
        curmrr=mrr(M,b,li)
        if bestmrr<curmrr:
            bestmrr=curmrr
            bestM=M
            bestb=b
    mpl.pyplot.plot(plot)
    return(bestM,bestb)

def loss(labels,q,M,a,b):
    """#Loss cross-entropy function with regularization
    inputs: labels-row array of {0.1};q-column vector;M-matrix;a-row of columns;b-scalar
    """
    x=-(labels*np.log(s.expit(z(q,M,a,b)))+(1-labels)*np.log(1-s.expit(z(q,M,a,b))))
    return np.sum(x)+l/2*(np.sum(M**2)+b**2)


#qTMa+b
def z(q,M,a,b):
    return np.dot(np.dot(np.transpose(q),M),a)+b

#Grad of loss over weights, 1 question 1 answer input
def grad(label,q,M,a,b):
    d=s.expit(z(q,M,a,b))-label
    gM=np.transpose(np.dot(a,q.reshape((1,50))))*d+l*M
    return (gM,d)

class yt(object):
    y=0
    t=0
    def __init__(self,y,t):
        self.y=y
        self.t=t

#Sorts probabilities and returns first True
def firstTrue(y,t):
    li=[]
    for i in range(0,len(y)):
        li.append(yt(y[i],t[i]))
    li.sort(key=lambda x: x.t,reverse=True)
    i=0
    for item in li:
        i+=1
        if item.y==1:
            return i
    return i+1

#Sum of losses for multiple qs
def lossAll(li,M,b):
    los=0
    for q in li:
        los+=loss(q.y,q.q,M,q.a,b)
    return los
    
#Returns MRR (used in uni)
def mrr(M,b,li):
    mrr=0.0
    for q in li:
        q.sett(M,b)
        mrr+=1/firstTrue(q.y,q.t)
    return mrr/len(li)

def setRes(li,ans1,ans0,res):
    p=0
    for i in range(0,len(li)):
        li[i].settcount(res[p:p+ans1[i]+ans0[i]])
        p+=ans1[i]+ans0[i]
    return    

#Returns MRR (used in uni+count)
def mrrcount(li,ans1,ans0):
    mrr=0.0
    for q in li:
        mrr+=1/firstTrue(q.y,q.tcount)
    return mrr/len(ans1)
    

def getInputs(li,ans1,ans0):
    y=np.zeros(sum(ans1)+sum(ans0))
    x=np.zeros((len(y),3))
    poz=0
    for q in li:
        for i in range(0,len(q.y)):
            x[poz+i]=[q.t[i],q.counts[i],q.idf[i]]
            y[poz+i]=q.y[i]
        poz+=len(q.y)
    return (x,y)

