import numpy as np
import pandas as pd
from HW1_ID3Algorithm import DT,ID3_algo,testtree,testID3,testing

data_names=list(pd.read_csv('/Users/u1368460/Documents/Machine Learning/ASSIGNMENTS/ASSIGNMENT 2/'
                            'car/data-desc.txt', skiprows=14))


trainData = pd.read_csv('/Users/u1368460/Documents/Machine Learning/ASSIGNMENTS/ASSIGNMENT 2/car/train.csv', names=data_names)

testData=pd.read_csv('/Users/u1368460/Documents/Machine Learning/ASSIGNMENTS/ASSIGNMENT 2/car/test.csv', names=data_names)

Onlyfeatures_train = np.array(trainData.iloc[:,:-1])
Onlyfeatures_test=np.array(testData.iloc[:,:-1])

featureNames = data_names[:-1]

Onlylabels_train=np.array(trainData.iloc[:,-1])
Onlylabels_test=np.array(testData.iloc[:,-1])

#Training
carTrain_Init = DT(trainData,m='entropy')
carTrain = ID3_algo(carTrain_Init)
#Testing
cartest1 = testtree(carTrain,testData,carTrain_Init)
error1,fullerror = testID3(cartest1)

#Output

m = ['entropy','majorityError','Gini']
tainingData=[Onlyfeatures_train,Onlylabels_train,trainData]
testingData = [Onlyfeatures_test,Onlylabels_test,testData]
_data_ = [trainData,testData]
deps = len(featureNames)

ei = testing(m,_data_,depth=deps)
traincar_error, testcar_error = testing.check(ei)

avg_car_train = np.mean(traincar_error,axis=1)
avg_car_test = np.mean(testcar_error,axis=1)

print('Errors in TRAINING :\n')
print('Method 1 : Entropy={}\n '.format(avg_car_train[0].round(3)))
print('Method 2: Majority Error={}\n'.format(avg_car_train[1].round(3)))
print('Method 3: Gini Index={}\n'.format(avg_car_train[2].round(3)))

print('Errors in TESTING :\n')
print('Method 1 : Entropy={}\n '.format(avg_car_test[0].round(3)))
print('Method 2 : Majority Error={}\n'.format(avg_car_test[1].round(3)))
print('Method 3:  Gini Index={}\n'.format(avg_car_test[2].round(3)))

