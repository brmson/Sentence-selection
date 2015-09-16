#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
input=jacana formated text files
output=pickled q objects
"""
from basicgrad import ttlist
from vecfromtext import textArrays,shortGlove
from bow import prepForGrad
import pickle
import numpy as np
from const import *
def saveQlist(QPATH,APATH1,APATH0,GLOVEPATH,GLOVEPATH2,PLIST,PANS1,PANS0,new_dict=False,c1=False,c0=False):
    """From jacana formated documents of questions, true answers, false answers
    saves list of Qs to PLIST path"""   
    (q,a1,a0,ans1,ans0)=textArrays(QPATH,APATH1,APATH0)
    if new_dict==True:
        shortGlove(q,a1,a0,GLOVEPATH,GLOVEPATH2)
    (qa,a1a,a0a)=prepForGrad(q,a1,a0,ans1,ans0,GLOVEPATH2)
    sentences=(q,a1,a0)
    li=ttlist(qa,a1a,a0a,ans1,ans0,sentences,c1,c0)
    pickle.dump( li, open( PLIST , "wb" ) )
    np.savetxt(PANS1,ans1)
    np.savetxt(PANS0,ans0)
    return
    
saveQlist(QPATH,APATH1,APATH0,GLOVEPATH,GLOVEPATH2,LISTPATH,PANS1,PANS0,new_dict=True,c1=CPATH1,c0=CPATH0)
print 'training data saved'
saveQlist(TQPATH,TAPATH1,TAPATH0,GLOVEPATH,TGLOVEPATH2,TLISTPATH,PTANS1,PTANS0,new_dict=True,c1=TCPATH1,c0=TCPATH0)
print 'testing data saved'
