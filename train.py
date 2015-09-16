#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
input=q objects
output=trained weights
"""

from basicgrad import mrrcount,mrr,setRes,getInputsClues,strictPercentage,testGrad
from const import *
import numpy as np
import pickle
from sklearn import linear_model
from vecfromtext import loadList,saveMb

def trecEval(li,count=True):
    truth=open('truth.txt','w')
    res=open('res.txt','w')
    for i in range(0,len(li)):
        for j in range(0,len(li[i].y)):
            truth.write(' '.join(map(str,(i,0,j,int(li[i].y[j]),'\n'))))
            if (count):
                res.write(' '.join(map(str,(i,0,j,1,li[i].tcount[j],'glove','\n'))))
            else:
                res.write(' '.join(map(str,(i,0,j,1,li[i].t[j],'glove','\n'))))
    truth.close()
    res.close()
    print 'trec_eval created'
    return
    
def train(LISTPATH,PANS1,PANS0,TLISTPATH,PTANS1,PTANS0):
    """Unigram+word count training from saved Qlist files, returns weights, generates trec_eval documents"""
    (trainlist,ans1,ans0)=loadList(LISTPATH,PANS1,PANS0)
    (testlist,tans1,tans0)=loadList(TLISTPATH,PTANS1,PTANS0)
    print 'data loaded'
    M=np.random.normal(0,0.01,(GLOVELEN,GLOVELEN))
    b=-0.0001
#    M=np.loadtxt('data/M58prop')
#    b=np.loadtxt('data/b58prop')
    (M,b)=testGrad(M,b,trainlist)



    x=0.0
    for i in range(0,len(tans1)):
        if (tans1[i]+tans0[i]>0):
            x+=float(tans1[i])/float(tans1[i]+tans0[i])
    print "random test mrr=",x/float(len(tans1))


    print 'MMR after unigram learning train:',mrr(M,b,trainlist)
    print 'MMR after unigram learning test:',mrr(M,b,testlist)

    pickle.dump((M, b), open("unigram-Mb.pickle", "wb"))

    # XXX: This has a sideeffect, setting resolutions in trainlist
    mrr(M,b,trainlist)

    (x,y)=getInputsClues(trainlist,ans1,ans0)
    print "train questions:",len(trainlist),
    print "train sentences:",len(x)
    (xtest,ytest)=getInputsClues(testlist,tans1,tans0)
    print "test questions:",len(testlist),
    print "test sentences:",len(xtest)
    
    
    
    print "% correct",100*float(sum(tans1))/float((sum(tans1)+sum(tans0)))    
    
    clf = linear_model.LogisticRegression(C=1, penalty='l2', tol=1e-5)
    clf.fit(x, y)
#    clf.coef_=np.array([[1,0.25]])
    tcounttest=clf.predict_proba(xtest)
    counttest=clf.predict_proba(x)
    setRes(trainlist,ans1,ans0,counttest[:,1])
    setRes(testlist,tans1,tans0,tcounttest[:,1])
    print 'MRR unigram+clues train',mrrcount(trainlist,ans1,ans0)
    print 'MRR unigram+clues test',mrrcount(testlist,tans1,tans0)
    
    print strictPercentage(testlist,tans1,tans0)*100,'% in the first three (test)'
    

    w=clf.coef_
    w=np.append(w,clf.intercept_);
    np.savetxt('data/weights.txt',w)
    return (M,b,w)



if __name__ == "__main__":
    # Seed always to the same number to get reproducible models
    np.random.seed(17151713)

    (M, b, w) = train(LISTPATH, PANS1, PANS0, TLISTPATH, PTANS1, PTANS0)

    saveMb(M,b,"data/Mbtemb.txt")
