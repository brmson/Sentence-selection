# -*- coding: utf-8 -*-
"""
Created on Wed Jul 08 18:15:54 2015

@author: Silvicek
"""
import numpy as np


qpath='data\jacana\Train1-100.Question.POSInput'
apath1='data\jacana\Train1-100.Positive-J.POSInput'
apath0='data\jacana\Train1-100.Negative-T.POSInput'
glovepath='data\glovewiki.txt'
glovepath2='usedembed.txt'

#Returns single GV from string
def getGloveVector(string):
    with open(glovepath2,'r') as f:
        for line in f:
            word=line.split(' ',1)[0]
            if word==string:
                x=np.array(line.split(' ')[1:]).astype(float)
                return x
    return None

#Returns qa vectors from files with jacana formating
def arraysFromQA():
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
    #            print x
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
    #            print x
            elif line[0:2]=='</':
                ans0.append(i)
                i=0
                

    return (questions,answers1,answers0,ans1,ans0)



#Creates smaller Glove-vector file with used words only
def shortGlove(questions,answers1,answers0):                
    i=0
    words=[]
    for sentence in questions:
        for word in questions[i]:
            if word not in words:
                words.append(word)
        i+=1
                
    i=0       
    for sentence in answers1:
        for word in answers1[i]:
            if word not in words:
                words.append(word)
        i+=1
    i=0       
    for sentence in answers0:
        for word in answers0[i]:
            if word not in words:
                words.append(word)
        i+=1
        
    
    used=open(glovepath2,'w')
    
    with open(glovepath,'r') as f:
        for line in f:
            word=line.split(' ',1)[0]
            if word in words:
    #            print 'found',word
                used.write(line)
                words.remove(word)
    used.close()
    return
    
    
    
    
def saveArrays(qa,a1a,a0a):
    np.savetxt('qarray.txt',qa)
    np.savetxt('a1rray.txt',a1a)
    np.savetxt('a0rray.txt',a0a)
    return

def loadArrays():
    qa=np.loadtxt('qarray.txt')
    a1a=np.loadtxt('a1rray.txt')
    a0a=np.loadtxt('a0rray.txt')
    return (qa,a1a,a0a)
    