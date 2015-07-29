# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:48:29 2015

@author: silvicek
"""
import os

PATH="data/java"
QPATH="data/jacana/Qtest.txt"
PPATH="data/jacana/Ptest.txt"
NPATH="data/jacana/Ntest.txt"
CPATH1="data/jacana/Clues1.txt"
CPATH0="data/jacana/Clues0.txt"

q=open(QPATH,'w')
p=open(PPATH,'w')
n=open(NPATH,'w')
cp=open(CPATH1,'w')
cn=open(CPATH0,'w')

qnum=0
for file in os.listdir(PATH):
    path=PATH+"/"+file
    i=0
    p.write("<A "+str(qnum)+">\n")
    n.write("<A "+str(qnum)+">\n")
    with open(path,'r') as f:
        for line in f:
            s=line.split(" ")
            if(s[0]=="<Q>" and i==0):
                q.write("<Q "+str(qnum)+">\n")
                q.write(" ".join(s[1:]))
                q.write("</Q>\n")
                i+=1
                qnum+=1
                continue
            elif(s[0]=="<Q>" and i!=0):
                continue
            if(s[0]=='1'):
                p.write(" ".join(s[3:]))
                cp.write(" ".join(s[1:3])+"\n")
            else:
                n.write(" ".join(s[3:]))
                cn.write(" ".join(s[1:3])+"\n")
    p.write("</A>\n")
    n.write("</A>\n")
    print "file done"
    
q.close()
p.close()
n.close()
c.close()

