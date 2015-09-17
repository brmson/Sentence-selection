#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
input=q objects
output=trained weights
"""
import time
import random
from basicgrad import mrrcount,mrr,setRes,getInputsClues,strictPercentage,testGrad
from const import *
import numpy as np
import pickle
from sklearn import linear_model
from vecfromtext import loadList,saveMb
from multiprocessing import Pool

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
  
def cross_validate_one(idx):
    global gdata
    (M,b,trainlist,threads)=gdata
    if idx==0:
        (M2,b2)=testGrad(M,b,trainlist,idx)
        res=0
    else:
        random.shuffle(trainlist)
        trainvalborder=len(trainlist)*(threads-2)/(threads-1)
        (M2,b2)=testGrad(M,b,trainlist[:trainvalborder],idx)
        print 'MMR after unigram learning train(idx=',idx,'):',mrr(M2,b2,trainlist)
        res=mrr(M2,b2,trainlist[trainvalborder:])
        print 'MMR after unigram learning val(idx=',idx,'):',res
    return (res,M2,b2)
    
def cross_validate_all(M,b,trainlist):
    global gdata
    threads=5
    gdata=(M,b,trainlist,threads+1)
    crossvalmrr=0
    i=0
    pool = Pool()
    for res in pool.imap(cross_validate_one,range(threads+1)):
        mrr,M,b=res
        crossvalmrr+=mrr
        if i==0:
            retM=M
            retb=b
            i+=1
    pool.close()
    return (crossvalmrr/threads,retM,retb)

def train(LISTPATH,PANS1,PANS0,TLISTPATH,PTANS1,PTANS0):
    """Unigram+word count training from saved Qlist files, returns weights, generates trec_eval documents"""
    (trainlist,ans1,ans0)=loadList(LISTPATH,PANS1,PANS0)
#    (testlist,tans1,tans0)=loadList(TLISTPATH,PTANS1,PTANS0)
    print 'data loaded'
    M=np.random.normal(0,0.01,(GLOVELEN,GLOVELEN))
    b=-0.0001
#    M=np.loadtxt('data/M58prop')
#    b=np.loadtxt('data/b58prop')
    t0=time.time()
    crossmrr,M,b=cross_validate_all(M,b,trainlist)
    t1=time.time()
    print "TIME =",t1-t0
    print "MRR after crossvalidation=",crossmrr

#    x=0.0
#    for i in range(0,len(tans1)):
#        if (tans1[i]+tans0[i]>0):
#            x+=float(tans1[i])/float(tans1[i]+tans0[i])
#    print "random test mrr=",x/float(len(tans1))


    pickle.dump((M, b), open("unigram-Mb.pickle", "wb"))

    # XXX: This has a sideeffect, setting resolutions in trainlist
    print('Mb MRR on train:', mrr(M,b,trainlist))
#    mrr(M,b,testlist)

    (x,y)=getInputsClues(trainlist,ans1,ans0)
    trainq=len(trainlist)
    trains=len(x)
    print "train questions:",trainq
    print "train sentences:",trains
#    (xtest,ytest)=getInputsClues(testlist,tans1,tans0)
#    valq=len(testlist)
#    vals=len(xtest)
#    print "test questions:",valq
#    print "test sentences:",vals
    
    
    
#    print "% correct",100*float(sum(tans1))/float((sum(tans1)+sum(tans0)))    
    
    clf = linear_model.LogisticRegression(C=1, penalty='l2', tol=1e-5)
    clf.fit(x, y)
#    clf.coef_=np.array([[1,0.25]])
#    tcounttest=clf.predict_proba(xtest)
    counttest=clf.predict_proba(x)
    setRes(trainlist,ans1,ans0,counttest[:,1])
#    setRes(testlist,tans1,tans0,tcounttest[:,1])
    mrrt=mrrcount(trainlist,ans1,ans0)
#    mrrv=mrrcount(testlist,tans1,tans0)
    print 'MRR unigram+clues train',mrrt
#    print 'MRR unigram+clues val',mrrv
    
#    print strictPercentage(testlist,tans1,tans0)*100,'% in the first three (test)'
    

    w=clf.coef_
    w=np.append(w,clf.intercept_);
    np.savetxt('data/weights.txt',w)
    results=[trainq,trains,mrrt]
    return (M,b,w,results)



if __name__ == "__main__":
    # Seed always to the same number to get reproducible models
    np.random.seed(17151713)

    (M, b, w, results) = train(LISTPATH, PANS1, PANS0, TLISTPATH, PTANS1, PTANS0)

    saveMb(M,b,"data/Mbtemb.txt",results)
