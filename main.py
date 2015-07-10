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


#mrr=bcmrrAll(q,a1,a0,ans1,ans0)

#print mrr


#Loss cross-entropy function with regularization
#inputs: labels-row array of {0.1};q-column vector;M-matrix;a-row of columns;b-scalar
def loss(labels,q,M,a,b):
    x=-(labels*np.log(s.expit(z(q,M,a,b)))+(1-labels)*np.log(1-s.expit(z(q,M,a,b))))
    return np.sum(x)+l/2*(np.sum(M**2)+b**2)




def z(q,M,a,b):
    return np.dot(np.dot(np.transpose(q),M),a)+b

#Grad of loss over weights, 1 question 1 answer input
def grad(label,q,M,a,b):
    d=s.expit(z(q,M,a,b))-label
    gM=np.transpose(np.dot(a,q.reshape((1,50))))*d+l*M
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


class q(object):
    q=[]
    a=[]
    y=[]
    t=[]
    def __init__(self,q,a1,a0):
        self.q=np.transpose(np.array(q,ndmin=2))
        a1=np.array(a1,ndmin=2)
        a0=np.array(a0,ndmin=2)
        self.a=np.hstack((np.transpose(a1),np.transpose(a0)))
#        print 'a1:',a1.shape,'a0:',a0.shape,'a:',self.a.shape
#        print 'newa:',self.a[:,0]
        self.y=np.hstack((np.ones(len(a1)),np.zeros(len(a0))))
#        self.t=np.zeros(len(self.y))
#        print 'ylen',self.y.shape
    def sett(self,M,b):
        self.t=z(self.q,M,self.a,b)

#print firstTrue(ytest,f(qtest,M,atest,b))


#qtest=qa[0].reshape((50,1))
#atest=np.transpose(np.vstack((a1a[:ans1[0]][:],a0a[:ans0[0]][:])))
#ytest=np.hstack((np.ones(ans1[0]),np.zeros(ans0[0])))

#qtest=qa[0:3].reshape((50,-1))
#atest=np.transpose(np.vstack((a1a[:ans1[0]][:],a0a[:ans0[0]][:])))
#ytest=np.hstack((np.ones(ans1[0]),np.zeros(ans0[0])))


def ttlists(qa,a1a,a0a,ans1,ans0):
    li=[]
    ones=0
    zeros=0
    for i in range(0,70):
    #    print a1a[ones]
        li.append(q(qa[i],a1a[ones:ones+ans1[i]],a0a[zeros:zeros+ans0[i]]))
    #    print 'q',li[0].q
        ones+=ans1[i]
        zeros+=ans0[i]
    testli=[]
    ones=0
    zeros=0
    for i in range(70,94):
    #    print a1a[ones]
        testli.append(q(qa[i],a1a[ones:ones+ans1[i]],a0a[zeros:zeros+ans0[i]]))
    #    print 'q',li[0].q
        ones+=ans1[i]
        zeros+=ans0[i]
    return(li,testli)

#np.savetxt('M70basicgrad.txt',M)
#np.savetxt('b70basicgrad.txt',b)


def lossAll(li,M,b):
    los=0
    for q in li:
        los+=loss(q.y,q.q,M,q.a,b)
#        print 'lossall',los
        
    return los





def mrr(M,b,li):
    mrr=0.0
    for q in li:
        q.sett(M,b)
        mrr+=1/firstTrue(q.y,q.t)
    return mrr/len(li)
    




def testGrad(M,b,li):
    alpha=2e-5
    plot=np.zeros(100)
    for i in range (0,1000):
        ggM=0
        ggb=0
        
        if i%10==0:
            plot[i/10]=lossAll(li,M,b)
            print plot[i/10]
            
        for q in li:
            for j in range(0,len(q.y)):
                (gM,gb)=grad(q.y[j],q.q,M,np.transpose(np.array(q.a[:,j],ndmin=2)),b)
                ggM+=gM
                ggb+=gb
            
            
        M=M-alpha*ggM
        b=b-alpha*ggb
    
    mpl.pyplot.plot(plot)
    return(M,b)
    
#################################################    
(quest,a1,a0,ans1,ans0)=arraysFromQA()
(qa,a1a,a0a)=loadArrays()
(trainlist,testlist)=ttlists(qa,a1a,a0a,ans1,ans0)
    
M=np.random.normal(0,0.01,(50,50))
b=0.00001    
b=np.loadtxt('b70basicgrad.txt')
M=np.loadtxt('M70basicgrad.txt')
l=1e-3
#(M,b)=testGrad(M,b,trainlist)
print 'mrr:',mrr(M,b,testlist)
#################################################








#def testGrad():
#    M=np.random.normal(0,0.01,(50,50))
#    alpha=4e-2
#    b=0
#    plot=np.zeros(100)
#    for i in range (0,1000):
#        ggM=0
#        ggb=0
#        
#        if i%10==0:
#            plot[i/10]=loss(ytest,qtest,M,atest,b)
#            print plot[i/10]
#        for j in range(0,33):
#            (gM,gb)=grad(ytest[j],qtest,M,atest[:,j].reshape((50,1)),b)
#            ggM+=gM
#            ggb+=gb
#            
#        M=M-alpha*ggM
#        b=b-alpha*ggb
#    
#    mpl.pyplot.plot(plot)
#    return
    

        




















#saveArrays(qa,a1a,a0a)
#(qa,a1a,a0a)=loadArrays()