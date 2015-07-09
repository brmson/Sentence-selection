cosdist mrr - spustit v main.py:
(q,a1,a0,ans1,ans0)=arraysFromQA()
mrr=bcmrrAll(q,a1,a0,ans1,ans0)
-trva cca 20min

zakladni gradient test spustit v main.py:
(q,a1,a0,ans1,ans0)=arraysFromQA()
(qa,a1a,a0a)=loadArrays()
qtest=qa[0].reshape((50,1))
atest=np.transpose(np.vstack((a1a[:ans1[0]][:],a0a[:ans0[0]][:])))
ytest=np.hstack((np.ones(ans1[0]),np.zeros(ans0[0])))
testGrad()
-prubezne vypisuje loss funkci