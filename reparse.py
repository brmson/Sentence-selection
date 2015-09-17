# -*- coding: utf-8 -*-
"""
Usage: reparse.py DATAPATH

input=yodaqa csv outputs (sentences)
output=jacana formated files for use in save.py
"""

import os
import sys

QPATH="data/Qtrain.txt"
PPATH="data/Ptrain.txt"
NPATH="data/Ntrain.txt"
CPATH1="data/Clues1train.txt"
CPATH0="data/Clues0train.txt"
TPATH="data/curated-test"
TQPATH="data/Qtest.txt"
TPPATH="data/Ptest.txt"
TNPATH="data/Ntest.txt"
TCPATH1="data/Clues1test.txt"
TCPATH0="data/Clues0test.txt"


def reparse(PATH,QPATH,PPATH,NPATH,CPATH1,CPATH0):
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
        print ".",
    q.close()
    p.close()
    n.close()
    cp.close()
    cn.close()


PATH = sys.argv[1]
reparse(PATH,QPATH,PPATH,NPATH,CPATH1,CPATH0)
reparse(TPATH,TQPATH,TPPATH,TNPATH,TCPATH1,TCPATH0)
