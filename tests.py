# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:41:00 2015

@author: silvicek
"""

from vecfromtext import shortGlove,textArrays
from basicgrad import ttlist,getInputs,mrrcount,mrr
from bow import prepForGrad
from const import *
import numpy as np
import pickle
from sklearn import linear_model
from io import saveQlist



##==================load=================
ans1=np.loadtxt(PANS1).astype(int)
ans0=np.loadtxt(PANS0).astype(int)
tans1=np.loadtxt(PTANS1).astype(int)
tans0=np.loadtxt(PTANS0).astype(int)
trainlist = pickle.load( open( LISTPATH, "rb" ) )
testlist = pickle.load( open( TLISTPATH, "rb" ) )
print 'data loaded'
##=======================================

#=================unigram===============
b=np.loadtxt('data/b64.txt')
M=np.loadtxt('data/M64.txt')
print 'best MRR unigram:',mrr(M,b,testlist)
#M=np.random.normal(0,0.01,(50,50))
#b=-0.0001
#print 'random',mrr(M,b,testlist)
#(M,b)=testGrad(M,b,trainlist,testlist)
b=np.loadtxt('data/b64.txt')
M=np.loadtxt('data/M64.txt')
#=======================================

#=============unigram+count=============
mrr(M,b,trainlist)
mrr(M,b,testlist)
(x,y)=getInputs(trainlist,ans1,ans0)
(xtest,ytest)=getInputs(testlist,tans1,tans0)
clf = linear_model.LogisticRegression(C=1, penalty='l2', tol=1e-5,solver='lbfgs')
clf.fit(x, y)
tcounttest=clf.predict_proba(xtest)
print 'MRR unigram+count',mrrcount(tcounttest[:,1],ytest,tans1,tans0)
#=======================================


#==============================================================================
# 

#def trainFromScratch(QPATH,APATH1,APATH0,GLOVEPATH2,new_dict=False):
#    M=np.random.normal(0,0.01,(50,50))
#    b=-0.0001
#    (M,b)=testGrad(M,b,li)
##    print 'MMR after learning:',mrr(M,b,li)
#    clf = linear_model.LogisticRegression(C=100, penalty='l2', tol=1e-5,solver='lbfgs')
#    clf.fit(x, y)
#    tcounttest=clf.predict_proba(xtest)
    
#    print 'MRR unigram+count',mrrcount(tcounttest[:,1],ytest,tans1,tans0)
    
    
#(q,a1,a0,ans1,ans0)=textArrays(QPATH,APATH1,APATH0)
#(tq,ta1,ta0,tans1,tans0)=textArrays(TQPATH,TAPATH1,TAPATH0)
#(qa,a1a,a0a)=prepForGrad(q,a1,a0,ans1,ans0,GLOVEPATH2)
#(tqa,ta1a,ta0a)=prepForGrad(tq,ta1,ta0,tans1,tans0,TGLOVEPATH2)
#sentences=(q,a1,a0)
#trainlist=ttlist(qa,a1a,a0a,ans1,ans0,sentences)
#tsentences=(tq,ta1,ta0)
#testlist=ttlist(tqa,ta1a,ta0a,tans1,tans0,tsentences)

#
#(trainlist,ans1,ans0)=loadQAlist(QPATH,APATH1,APATH0,GLOVEPATH2)
#(testlist,tans1,tans0)=loadQAlist(TQPATH,TAPATH1,TAPATH0,TGLOVEPATH2)

#b=np.loadtxt('data/b64.txt')
#M=np.loadtxt('data/M64.txt')
#mrr(M,b,trainlist)
#mrr(M,b,testlist)

#print 'lists loaded'
#==============================================================================


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


#==============================================================================
