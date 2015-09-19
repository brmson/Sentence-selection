#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
input=q objects
output=trained weights
"""

import time
import random
from basicgrad import mrrcount,mrr,setRes,getInputsClues,testGrad,trainConsts
from const import *
import numpy as np
from sklearn import linear_model
from vecfromtext import loadList,saveMb
from multiprocessing import Pool
  
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
    i=0
    pool = Pool()
    mrrs=[]
    for res in pool.imap(cross_validate_one,range(threads+1)):
        mrr,M,b=res
        if i==0:
            retM=M
            retb=b
            i+=1
        else:
            mrrs.append(mrr)
    pool.close()
    return (mrrs,sum(mrrs)/threads,retM,retb)

def trainMb(trainlist,ans1,ans0):
    """Unigram training from saved Qlist files, returns Mb weights.
    You can play with the learning constants in trainConsts() of basicgrad.py"""
    t0=time.time()
    M=np.random.normal(0,0.01,(GLOVELEN,GLOVELEN))
    b=-0.0001
#    M=np.loadtxt('data/M58prop')
#    b=np.loadtxt('data/b58prop')
    mrrs,crossmrr,M,b=cross_validate_all(M,b,trainlist)
    t1=time.time()
    print "time spent training =",t1-t0
    print "MRR after crossvalidation=",crossmrr

    # XXX: This has a sideeffect, setting resolutions in trainlist
    trainmrr=mrr(M,b,trainlist)
    print 'Mb MRR on train:', trainmrr
    l,alpha=trainConsts()
    results=[crossmrr,mrrs,l,alpha,trainmrr]
    return (M,b,results)


def trainClues(trainlist,ans1,ans0):
    """Logistic regression using Mb probability and clues as input.
    requires mrr(M,b,trainlist) called beforehand to work properly"""
    (x,y)=getInputsClues(trainlist,ans1,ans0)
    clf = linear_model.LogisticRegression(C=1, penalty='l2', tol=1e-5)
    clf.fit(x, y)
    counttest=clf.predict_proba(x)
    setRes(trainlist,ans1,ans0,counttest[:,1])
    mrrt=mrrcount(trainlist,ans1,ans0)
    print 'MRR unigram+clues train',mrrt
    w=clf.coef_
    w=np.append(w,clf.intercept_);
    return w

def train(LISTPATH,PANS1,PANS0):
    (trainlist,ans1,ans0)=loadList(LISTPATH,PANS1,PANS0)
    print 'data loaded'
    (M,b,results)=trainMb(trainlist,ans1,ans0)
    w=trainClues(trainlist,ans1,ans0)
    
    prop_num=0
    for q in trainlist:
        prop_num+=len(q.y)
    q_num=len(trainlist)
    print "trained on",q_num,"questions"
    print "trained on",prop_num,"properties"
    crossmrr,mrrs,l,alpha,trainmrr=results
    results=(q_num,prop_num,crossmrr,mrrs,l,alpha,trainmrr)
    return (M,b,w,results)
    

if __name__ == "__main__":
    # Seed always to the same number to get reproducible models
    np.random.seed(17151713)

    (M, b, w, results) = train(LISTPATH, PANS1, PANS0)

    saveMb(M,b,"data/Mbtemp.txt",results)
    np.savetxt('data/weights.txt',w)