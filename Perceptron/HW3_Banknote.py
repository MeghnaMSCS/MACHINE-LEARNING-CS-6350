import numpy as np
import pandas as pd
from HW3_Perceptron import perceptronAlgo

data_names = ['variance','skewness','curtosis','entropy','label']

trainData = pd.read_csv('/Users/u1368460/Documents/Machine Learning/HW3/bank-note/train.csv', names=data_names)

testData=pd.read_csv('/Users/u1368460/Documents/Machine Learning/HW3/bank-note/test.csv', names=data_names)

T=10
rate=0.01
ini_p = perceptronAlgo(T,rate,trainData)
trainedpAlgo = perceptronAlgo.trainpAlgo(ini_p)
error_std = perceptronAlgo.Pprediction(trainedpAlgo,testData)

ini_v = perceptronAlgo(T,rate,trainData,type='vote')
trainedVoteAlgo = perceptronAlgo.trainVoteAlgo(ini_v)
error_vote = perceptronAlgo.Pprediction(trainedVoteAlgo,testData)

ini_avg = perceptronAlgo(T,rate,trainData,type='average')
trainedAvgAlgo = perceptronAlgo.trainAvgAlgo(ini_avg)
error_average = perceptronAlgo.Pprediction(trainedAvgAlgo,testData)


tr1,tr2,tr3 = [],[],[]

error1,error2,error3 = [],[],[]

print('Training 100 Perceptrons from each type')
for j in range(100):
    ini_p=perceptronAlgo(T,rate,trainData)
    trainedpAlgo=perceptronAlgo.trainpAlgo(ini_p)
    tr1.append(perceptronAlgo.Pprediction(trainedpAlgo,trainData))
    error1.append(perceptronAlgo.Pprediction(trainedpAlgo,testData))

    ini_v=perceptronAlgo(T,rate,trainData,type='vote')
    trainedVoteAlgo=perceptronAlgo.trainVoteAlgo(ini_v)
    tr2.append(perceptronAlgo.Pprediction(trainedVoteAlgo,trainData))
    error2.append(perceptronAlgo.Pprediction(trainedVoteAlgo,trainData))

    ini_avg=perceptronAlgo(T,rate,trainData,type='average')
    trainedAvgAlgo=perceptronAlgo.trainAvgAlgo(ini_avg)
    tr3.append(perceptronAlgo.Pprediction(trainedAvgAlgo,trainData))
    error3.append(perceptronAlgo.Pprediction(trainedAvgAlgo,testData))
    print('Training in progress')
print('Finished')


output = f""" Here is the output
Type = Standard
Weight vector = {trainedpAlgo.wv}
Error in prediction = {error_std}

Type = Voted
Total Weight vectors = {len(trainedVoteAlgo.w_v)}
Part upto 50 examples = {trainedVoteAlgo.w_v[49]} \t\t
Part upto 100 examples = {trainedVoteAlgo.w_v[99]} \t\t
Last Weight vector = {trainedVoteAlgo.wv}
Error in prediction = {error_vote}  

Type = Averaged
Weight vector = {trainedAvgAlgo.wv}  
Error in prediction = {error_average}    

"""

print(output)

