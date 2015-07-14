# -*- coding: utf-8 -*-
"""
Created on Wed Jul 08 18:15:54 2015

@author: Silvicek
"""
import numpy as np

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
#Returns single GV from string
def getGloveVector(string,glovepath2):
    with open(glovepath2,'r') as f:
        for line in f:
            word=line.split(' ',1)[0]
            if word==string:
                x=np.array(line.split(' ')[1:]).astype(float)
                return x
    return None

#Returns qa vectors from files with jacana formating
def textArrays(qpath,apath1,apath0):
    questions=[]
    with open(qpath,'r') as f:
        for line in f:
            line=line.lower()
            if line[0]!='<':
                x=np.array(line.split(' ')[:-1])
                questions.append(x)
                
    answers1=[]
    i=0
    ans1=[]
    with open(apath1,'r') as f:
        for line in f:
            line=line.lower()
            if line[0]!='<':
                i+=1
                x=np.array(line.split(' ')[:-1])
                answers1.append(x)
            elif line[0:2]=='</':
                ans1.append(i)
                i=0
    answers0=[]
    i=0
    ans0=[]
    with open(apath0,'r') as f:
        for line in f:
            line=line.lower()
            if line[0]!='<':
                i+=1
                x=np.array(line.split(' ')[:-1])
                if len(x)<1:
                    i-=1
                else:
                    answers0.append(x)
            elif line[0:2]=='</':
                ans0.append(i)
                i=0
    return (questions,answers1,answers0,ans1,ans0)

#Creates smaller Glove-vector file with used words only
def shortGlove(questions,answers1,answers0,glovepath2):                
    i=0
    words=set()
    for sentence in questions:
        for word in questions[i]:
            if word not in words:
                words.add(word)
        i+=1
    i=0       
    for sentence in answers1:
        for word in answers1[i]:
            if word not in words:
                words.add(word)
        i+=1
    i=0       
    for sentence in answers0:
        for word in answers0[i]:
            if word not in words:
                words.add(word)
        i+=1
    used=open(glovepath2,'w')
    with open(GLOVEPATH,'r') as f:
        for line in f:
            word=line.split(' ',1)[0]
            if word in words:
                print 'found',word
                used.write(line)
                words.remove(word)
    used.close()
    return

def saveArrays(qa,a1a,a0a,ans1,ans0,pqa,pa1a,pa0a,pans1,pans0):
    np.savetxt(pqa,qa)
    np.savetxt(pa1a,a1a)
    np.savetxt(pa0a,a0a)
    np.savetxt(pans1,ans1)
    np.savetxt(pans0,ans0)
    return
    
def loadArrays(qa,a1a,a0a):
    qa=np.loadtxt(qa)
    a1a=np.loadtxt(a1a)
    a0a=np.loadtxt(a0a)
    return (qa,a1a,a0a)
#def loadArrays(qa,a1a,a0a,ans1,ans0):
#    qa=np.loadtxt(qa)
#    a1a=np.loadtxt(a1a)
#    a0a=np.loadtxt(a0a)
#    ans1=np.loadtxt(ans1)
#    ans0=np.loadtxt(ans0)
#    return (qa,a1a,a0a,ans1,ans0)
    