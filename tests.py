# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:41:00 2015

@author: silvicek
"""

from vecfromtext import *
from bow import *
from basicgrad import *
import numpy as np
import pickle

TQPATH='data/jacana/Test.Question.POSInput'
TAPATH1='data/jacana/Test.Positive-J.POSInput'
TAPATH0='data/jacana/Test.Negative-T.POSInput'
QPATH='data/jacana/Train1-100.Question.POSInput'
APATH1='data/jacana/Train1-100.Positive-J.POSInput'
APATH0='data/jacana/Train1-100.Negative-T.POSInput'
GLOVEPATH='data/glovewiki.txt'
GLOVEPATH2='data/tusedembed.txt'
PTQA='data/tqarray.txt'
PTA1A='data/ta1rray.txt'
PTA0A='data/ta0rray.txt'
PTANS1='data/tans1.txt'
PTANS0='data/tans0.txt'
PQA='data/qarray.txt'
PA1A='data/a1rray.txt'
PA0A='data/a0rray.txt'
PANS1='data/ans1.txt'
PANS0='data/ans0.txt'

def countWords(trainlist,testlist,ans1,ans0,tans1,tans0):
    y=np.zeros(sum(ans1)+sum(ans0))
    x=np.zeros((len(y),3))
    poz=0
    for q in trainlist:
        for i in range(0,len(q.y)):
            x[poz+i]=[q.t[i],q.counts[i],q.idf[i]]
            y[poz+i]=q.y[i]
        poz+=len(q.y)
    ytest=np.zeros(sum(tans1)+sum(tans0))
    xtest=np.zeros((len(ytest),3))
    poz=0
    for q in testlist:
        for i in range(0,len(q.y)):
            xtest[poz+i]=[q.t[i],q.counts[i],q.idf[i]]
            ytest[poz+i]=q.y[i]
        poz+=len(q.y)
    return
#==================load=================
ans1=np.loadtxt(PANS1).astype(int)
ans0=np.loadtxt(PANS0).astype(int)
tans1=np.loadtxt(PTANS1).astype(int)
tans0=np.loadtxt(PTANS0).astype(int)
trainlist = pickle.load( open( "data/trainlist.p", "rb" ) )
testlist = pickle.load( open( "data/testlist.p", "rb" ) )
countWords(trainlist,testlist,ans1,ans0,tans1,tans0)
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




#############################################
#np.savetxt('data/tqarray.txt',qa)
#np.savetxt('data/ta1rray.txt',a1a)
#np.savetxt('data/ta0rray.txt',a0a)
#np.savetxt('data/tans1.txt',tans1)
#np.savetxt('data/tans0.txt',tans0)
#np.savetxt('data/M64.txt',M)
#np.savetxt('data/b64.txt',b)

#(tquestions,tanswers1,tanswers0,tans1,tans0)=arraysFromQA(QPATH,APATH1,APATH0)
#shortGlove(tquestions,tanswers1,tanswers0,GLOVEPATH2)
#(qa,a1a,a0a)=prepForGrad(tquestions,tanswers1,tanswers0,tans1,tans0,GLOVEPATH2)

#(qa,a1a,a0a)=loadArrays(PQA,PA1A,PA0A)
#(tqa,ta1a,ta0a)=loadArrays(PTQA,PTA1A,PTA0A)
##
##
#sentences=(questions,answers1,answers0)
#trainlist=ttlist(qa,a1a,a0a,ans1,ans0,sentences)
#tsentences=(tquestions,tanswers1,tanswers0)
#testlist=ttlist(tqa,ta1a,ta0a,tans1,tans0,tsentences)
#print 'lists loaded'

#pickle.dump( trainlist, open( "data/trainlist.p", "wb" ) )
#pickle.dump( testlist, open( "data/testlist.p", "wb" ) )


#(tquestions,tanswers1,tanswers0,tans1,tans0)=textArrays(TQPATH,TAPATH1,TAPATH0)
#(questions,answers1,answers0,ans1,ans0)=textArrays(QPATH,APATH1,APATH0)