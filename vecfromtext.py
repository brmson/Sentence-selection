# -*- coding: utf-8 -*-

import numpy as np
import pickle
import re

def getGloveDict(glovepath2):
    """Returns discionary of used words"""
    gloveDict = dict()
    with open(glovepath2,'r') as f:
        for line in f:
            word=line.split(' ',1)[0]
            gloveDict[word] = np.array(line.split(' ')[1:]).astype(float)
    return gloveDict


def textArrays(qpath,apath1,apath0):
    """ Returns qa text vectors from files with jacana formating.
    Text == array of tokens.
    It is a tuple of:
      * a list of question texts
      * a list of texts of all correct answers (across all questions)
      * a list of texts of all incorrect answers
      * for each question, #of correct answers (used for computing the index in list of all correct answers)
      * for each question, #of incorrect answers
    """
    questions=[]
    with open(qpath,'r') as f:
        for line in f:
            line=line.lower()
            if line[0]!='<':
                line=re.sub('[^0-9a-zA-Z]+',' ', line)
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
                line=re.sub('[^0-9a-zA-Z]+',' ', line)
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
                line=re.sub('[^0-9a-zA-Z]+',' ', line)
                x=np.array(line.split(' ')[:-1])
                if len(x)<1:
                    i-=1
                else:
                    answers0.append(x)
            elif line[0:2]=='</':
                ans0.append(i)
                i=0
    return (questions,answers1,answers0,ans1,ans0)

def shortGlove(questions,answers1,answers0,glovepath_in,glovepath_out):                
    """ From a full Glove dictionary (glovepath2),
    creates smaller Glove-vector file with used words only """
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
    used=open(glovepath_out,'w')
    with open(glovepath_in,'r') as f:
        for line in f:
            word=line.split(' ',1)[0]
            if word in words:
#                print 'found',word
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
    
#    results=[trainq,trains,mrrt]
def saveMb(M,b,path,results):
    np.savetxt(path,M)
    m=open(path,'a')
    m.write("%f"%b)
    m.close()
    with open(path, "r+") as f:
         old = f.read() # read everything in the file
         f.seek(0) # rewind
         f.write("\\\\Weights for property-selection feature, generated with https://github.com/brmson/Sentence-selection\n" +
         "\\\\Trained on "+str(results[0])+" questions, "+str(results[1])+" properties\n"+
#         "\\\\Validated on "+str(results[2])+" questions, "+str(results[3])+" properties\n"+
         "\\\\trainMRR = "+str(results[2])+"\n" 
#         "\\\\valMRR = "+str(results[5])+"\n"
         +old) # write the new line before
    
def loadArrays(qa,a1a,a0a):
    qa=np.loadtxt(qa)
    a1a=np.loadtxt(a1a)
    a0a=np.loadtxt(a0a)
    return (qa,a1a,a0a) 
    
def loadList(LISTPATH,PANS1,PANS0):
    ans1=np.loadtxt(PANS1).astype(int)
    ans0=np.loadtxt(PANS0).astype(int)
    li = pickle.load( open( LISTPATH, "rb" ) )
    return (li,ans1,ans0)