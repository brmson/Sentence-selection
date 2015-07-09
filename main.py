# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 22:27:33 2015

@author: Silvicek
"""

import numpy as np
#import scipy as sp
#from sklearn.metrics import log_loss

import scipy.special as s

from vecfromtext import arraysFromQA,loadArrays
from bow import bcmrrAll

import matplotlib as mpl

#(q,a1,a0,ans1,ans0)=arraysFromQA()


#mrr=bcmrrAll(q,a1,a0,ans1,ans0)

#print mrr


#Loss cross-entropy function with regularization
#inputs: labels-row array of {0.1};q-column vector;M-matrix;a-row of columns;b-scalar
def loss(labels,q,M,a,b):
    x=-(labels*np.log(s.expit(z(q,M,a,b)))+(1-labels)*np.log(1-s.expit(z(q,M,a,b))))
    return np.sum(x)+l/2*(np.sum(M**2)+b**2)




def z(q,M,a,b):
    return np.dot(np.dot(np.transpose(q),M),a)+b

#Grad of loss over weights, 1 question n answers input
def grad(label,q,M,a,b):
    d=s.expit(z(q,M,a,b))-label
    gM=np.transpose(np.dot(a.reshape((50,1)),q.reshape((1,50))))*d
    return (gM,d)

def f(q,M,a,b):
    return s.expit(z(q,M,a,b))




class yt(object):
    y=0
    t=0
    def __init__(self,y,t):
        self.y=y
        self.t=t
        
def firstTrue(y,t):
    li=[]
    for i in range(0,len(y)):
        li.append(yt(y[i],t[0][i]))
    li.sort(key=lambda x: x.t,reverse=True)
    i=0
    for item in li:
        i+=1
        if item.y==1:
            return i
    return i+1
    
print firstTrue(ytest,f(qtest,M,atest,b))
  
#(qa,a1a,a0a)=loadArrays()
#qtest=qa[0].reshape((50,1))
#atest=np.transpose(np.vstack((a1a[:ans1[0]][:],a0a[:ans0[0]][:])))
#ytest=np.hstack((np.ones(ans1[0]),np.zeros(ans0[0])))



def testGrad():
    l=0
    M=np.random.normal(0,0.01,(50,50))
    alpha=4e-2
    b=0
    plot=np.zeros(100)
    for i in range (0,10000):
        ggM=0
        ggb=0
        
        if i%100==0:
            print loss(ytest,qtest,M,atest,b)
            plot[i/100]=loss(ytest,qtest,M,atest,b)
        for j in range(0,33):
            (gM,gb)=grad(ytest[j],qtest,M,atest[:,j].reshape((50,1)),b)
            ggM+=gM
            ggb+=gb
            
        M=M-alpha*ggM
        b=b-alpha*ggb
    
    mpl.pyplot.plot(plot)
    return
    
#testGrad()
        




















#saveArrays(qa,a1a,a0a)
#(qa,a1a,a0a)=loadArrays()