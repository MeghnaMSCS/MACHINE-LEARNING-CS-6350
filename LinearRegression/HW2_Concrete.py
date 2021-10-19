from HW2_Linearregression import methodLMS,callStochLMS,callBatchLMS
import numpy as np
import pandas as pd

data_names = ['Cement','Slag','Fly ash','Water','SP','Coarse Aggr','Fine Aggr', 'Output']

trainData = pd.read_csv('/Users/u1368460/Documents/Machine Learning/HW2/concrete/train.csv',names=data_names)
testData = pd.read_csv('/Users/u1368460/Documents/Machine Learning/HW2/concrete/test.csv',names=data_names)

l_r=0.0125
batch=methodLMS(trainData,5000,l_r)
batch_wt= callBatchLMS(batch)

aTest=np.append(np.ones((len(testData),1)), testData.iloc[:,:-1],1)
bTest = testData.iloc[:,-1]
JtestBatch = 0.5*sum((bTest-batch_wt.dot(aTest.T))**2)

tm = 50000
l_r2 = 0.0125
stoch = methodLMS(trainData,tm,l_r2)
stoch_wt = callStochLMS(stoch)


outputs = f""" 1) Batch Method: Rate of learning = {l_r} , Weight = {np.round(batch_wt,3)}
               2) Stochastic Method : Rate of learning = {l_r} , Weight = {np.round(stoch_wt,3)}
"""

print(outputs)

