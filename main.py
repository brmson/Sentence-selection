# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 22:27:33 2015

@author: Silvicek
"""

from vecfromtext import *
from bow import *
from basicgrad import *
import numpy as np
import pickle
from sklearn import linear_model


#==================load=================
ans1=np.loadtxt(PANS1).astype(int)
ans0=np.loadtxt(PANS0).astype(int)
tans1=np.loadtxt(PTANS1).astype(int)
tans0=np.loadtxt(PTANS0).astype(int)
trainlist = pickle.load( open( "data/trainlist.p", "rb" ) )
testlist = pickle.load( open( "data/testlist.p", "rb" ) )
(x,y,xtest,ytest)=countWords(trainlist,testlist,ans1,ans0,tans1,tans0)
print 'data loaded'
#=======================================


#==================load=================
ans1=np.loadtxt(PANS1).astype(int)
ans0=np.loadtxt(PANS0).astype(int)
tans1=np.loadtxt(PTANS1).astype(int)
tans0=np.loadtxt(PTANS0).astype(int)
trainlist = pickle.load( open( "data/trainlist.p", "rb" ) )
testlist = pickle.load( open( "data/testlist.p", "rb" ) )
(x,y,xtest,ytest)=countWords(trainlist,testlist,ans1,ans0,tans1,tans0)
print 'data loaded'
#=======================================

#=================unigram===============
b=np.loadtxt('data/b64.txt')
M=np.loadtxt('data/M64.txt')
print 'best MRR unigram:',mrr(M,b,testlist)
M=np.random.normal(0,0.01,(50,50))
b=-0.0001
(M,b)=testGrad(M,b,trainlist,testlist)
print 'MRR unigram:',mrr(M,b,testlist)
#=======================================

#=============unigram+count=============
clf = linear_model.LogisticRegression(C=100, penalty='l2', tol=1e-5)
clf.fit(x, y)
tcounttest=clf.predict_proba(xtest)
print 'MRR unigram+count',mrrcount(tcounttest[:,1],ytest,ans1,ans0)
#=======================================
