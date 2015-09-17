# -*- coding: utf-8 -*-
"""
Usage: reparseprops.py DATAPATH

input=yodaqa csv outputs (properties)
output=jacana formated files for use in save.py
"""

import os
import sys

QPATH="data/Qtrain.txt"
PPATH="data/Ptrain.txt"
NPATH="data/Ntrain.txt"
CPATH1="data/Clues1train.txt"
CPATH0="data/Clues0train.txt"
#TPATH="data/curated-test"
#TQPATH="data/Qtest.txt"
#TPPATH="data/Ptest.txt"
#TNPATH="data/Ntest.txt"
#TCPATH1="data/Clues1test.txt"
#TCPATH0="data/Clues0test.txt"

def notNumber(s):
    try:
        float(s)
        return False
    except ValueError:
        return True


def reparseProps(PATH,QPATH,PPATH,NPATH,CPATH1,CPATH0):
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
        propdict=dict()
        propset=set()
        with open(path,'r') as f:
            for line in f:
                s=line.split(" ")
                if(s[0]!="<Q>"):
                    s=line.split(" ")
                    text=" ".join(s[2:]).lower()
                    if text in propdict:
                        if(s[0]=='1'):
                            propdict[text]='1'
                        continue
                    propdict[text]=s[0]
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
                if notNumber(s[0]) or notNumber(s[1]):
                    continue
#                print s
                text=" ".join(s[2:]).lower()
                if text not in propset:
#                    print text
                    if(propdict[text]=='1'):
                        p.write(text)
                        cp.write(" ".join(s[1:2])+"\n")
                    else:
                        n.write(text)
                        cn.write(" ".join(s[1:2])+"\n")
                    propset.add(text)
        p.write("</A>\n")
        n.write("</A>\n")
        print ".",
    q.close()
    p.close()
    n.close()
    cp.close()
    cn.close()


PATH = sys.argv[1]
reparseProps(PATH,QPATH,PPATH,NPATH,CPATH1,CPATH0)
#reparseProps(TPATH,TQPATH,TPPATH,TNPATH,TCPATH1,TCPATH0)
