# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:17:03 2015

@author: silvicek
"""

from basicgrad import getInputs,mrrcount,mrr,testGrad
from const import *
import numpy as np
from sklearn import linear_model
from vecfromtext import loadList


def train(LISTPATH,PANS1,PANS0,TLISTPATH,PTANS1,PTANS0):
    """Unigram+word count training from saved Qlist files"""
    (trainlist,ans1,ans0)=loadList(LISTPATH,PANS1,PANS0)
    (testlist,tans1,tans0)=loadList(TLISTPATH,PTANS1,PTANS0)
    print 'data loaded'
    M=np.random.normal(0,0.01,(50,50))
    b=-0.0001
    (M,b)=testGrad(M,b,trainlist)
    print 'MMR after unigram learning:',mrr(M,b,testlist)
    mrr(M,b,trainlist)
    (x,y)=getInputs(trainlist,ans1,ans0)
    (xtest,ytest)=getInputs(testlist,tans1,tans0)
    clf = linear_model.LogisticRegression(C=100, penalty='l2', tol=1e-5,solver='lbfgs')
    clf.fit(x, y)
    tcounttest=clf.predict_proba(xtest)
    print 'MRR unigram+count',mrrcount(tcounttest[:,1],ytest,tans1,tans0)
    return (M,b,clf.get_params())

(M,b,w)=train(LISTPATH,PANS1,PANS0,TLISTPATH,PTANS1,PTANS0)