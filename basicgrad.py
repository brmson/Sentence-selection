# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 09:49:42 2015

@author: Silvicek
"""
import numpy as np
import matplotlib as mpl
import scipy.special as s


l=1e-5      #regularisation constant
alpha=2e-5  #learning constant


#Holds questions with all its answers and T/F values as well as counted probabilities
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
        self.y=np.hstack((np.ones(len(a1)),np.zeros(len(a0))))
    def sett(self,M,b):
        self.t=z(self.q,M,self.a,b)

#Returns train and test lists of qs
def ttlists(qa,a1a,a0a,ans1,ans0):
    li=[]
    ones=0
    zeros=0
    for i in range(0,70):
        li.append(q(qa[i],a1a[ones:ones+ans1[i]],a0a[zeros:zeros+ans0[i]]))
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
    

#Updates weights using basic gradient descent
def testGrad(M,b,li,tli):
    plot=np.zeros(100)
    for i in range (0,1000):
        ggM=0
        ggb=0
        if i%10==0:
            plot[i/10]=lossAll(li,M,b)
            print plot[i/10]
#            print 'mrr train:',mrr(M,b,li)
#            print 'mrr test:',mrr(M,b,tli)
        for q in li:
            for j in range(0,len(q.y)):
                (gM,gb)=grad(q.y[j],q.q,M,np.transpose(np.array(q.a[:,j],ndmin=2)),b)
                ggM+=gM
                ggb+=gb
        M=M-alpha*ggM
        b=b-alpha*ggb
    mpl.pyplot.plot(plot)
    return(M,b)
 
 
#Loss cross-entropy function with regularization
#inputs: labels-row array of {0.1};q-column vector;M-matrix;a-row of columns;b-scalar
def loss(labels,q,M,a,b):
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

#sigmoid
def f(q,M,a,b):
    return s.expit(z(q,M,a,b))


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
        li.append(yt(y[i],t[0][i]))
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




def mrr(M,b,li):
    mrr=0.0
    rand=0.0
    for q in li:
        q.sett(M,b)
        mrr+=1/firstTrue(q.y,q.t)
        rand+=sum(q.y)/len(q.y)
#        print firstTrue(q.y,q.t),'out of',len(q.y),'(',sum(q.y),')'
#    print 'random:',rand/len(li)
    return mrr/len(li)
    
