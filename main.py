# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 22:27:33 2015

@author: Silvicek
"""

import numpy as np
#import scipy as sp
#from sklearn.metrics import log_loss
#import matplotlib as mpl
#import scipy.special as s

from vecfromtext import arraysFromQA,loadArrays
from basicgrad import mrr,ttlists,testGrad


#(quest,a1,a0,ans1,ans0)=arraysFromQA()


#################################################    
(qa,a1a,a0a,ans1,ans0)=loadArrays()
(trainlist,testlist)=ttlists(qa,a1a,a0a,ans1,ans0)  
M=np.random.normal(0,0.01,(50,50))
b=-0.00001    
b=np.loadtxt('data/b70basicgrad.txt')
M=np.loadtxt('data/M70basicgrad.txt')

#(M,b)=testGrad(M,b,trainlist,testlist)
print 'mrr test:',mrr(M,b,testlist)
#################################################


#np.savetxt('data/M70basicgrad.txt',M)
#np.savetxt('data/b70basicgrad.txt',b)

