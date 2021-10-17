import numpy as np
import pandas as pd
from HW1_ID3Algorithm import DT,ID3_algo,testtree,testID3,testing

def unknown(data):
    u_col=data.columns
    for c in u_col:
        if isinstance(data[c][0], str):
            most_common=data[c].value_counts()
            indx=0
            if most_common.index[0]=='unknown':
                indx=1
            data[c]=data[c].replace('unknown',most_common.index[indx])
    return data


nametitles=['age','job','marital','education','default','balance','housing','loan','contact','day','month',
          'duration','campaign','pdays','previous','poutcome','y']


trainData = pd.read_csv('/Users/u1368460/Documents/Machine Learning/ASSIGNMENTS/ASSIGNMENT 2/bank/train.csv', names=nametitles)
testData=pd.read_csv('/Users/u1368460/Documents/Machine Learning/ASSIGNMENTS/ASSIGNMENT 2/bank/test.csv', names=nametitles)

withoutUnknown_train = unknown(trainData.copy())
withoutUnknown_test = unknown(testData.copy())

#Considering 'unknown' as an attribute value
bankTrain_Init = DT(trainData,num=True)
bankTrain = ID3_algo(bankTrain_Init)

banktest1 = testtree(bankTrain,testData,bankTrain_Init,num=True)
error1,fullerror = testID3(banktest1)

m = ['entropy','majorityError','Gini']
dep = len(trainData.columns) - 1
data_ = [trainData,testData]

banktest1=testing(m,data_,depth=dep,num=True)
trainbank_error1, testbank_error1 = testing.check(banktest1)

#Replacing 'unknown' with majority values

data_2 = [withoutUnknown_train, withoutUnknown_test]
banktest2=testing(m,data_2,depth=dep,num=True)
trainbank_error2, testbank_error2 = testing.check(banktest2)

#Output
#Considering 'unknown' as an attribute value
average_Tr=np.mean(trainbank_error1,axis=1)
average_Te=np.mean(testbank_error1, axis=1)

print('Errors in TRAINING :\n')
print('Method 1 : Entropy={}\n '.format(average_Tr[0].round(4)))
print('Method 2: Majority Error={}\n'.format(average_Tr[1].round(4)))
print('Method 3: Gini Index={}\n'.format(average_Tr[2].round(4)))

print('Errors in TESTING :\n')
print('Method 1 : Entropy={}\n '.format(average_Te[0].round(4)))
print('Method 2 : Majority Error={}\n'.format(average_Te[1].round(4)))
print('Method 3:  Gini Index={}\n'.format(average_Te[2].round(4)))

#Output
#Replacing 'unknown' with majority values

replaceUnk_avg_train = np.mean(trainbank_error2,axis=1)
replaceUnk_avg_test = np.mean(testbank_error2,axis=1)

print('Errors in TRAINING :\n')
print('Method 1 : Entropy={}\n'.format(replaceUnk_avg_train[0].round(4)))
print('Method 2 : Majority Error={}\n'.format(replaceUnk_avg_train[1].round(4)))
print('Method 3 : Gini Index={}\n '.format(replaceUnk_avg_train[2].round(4)))

print('Errors in TESTING :\n')
print('Method 1 : Entropy={}\n '.format(replaceUnk_avg_test[0].round(4)))
print('Method 2 : Majority Error={}\n'.format(replaceUnk_avg_test[1].round(4)))
print('Method 3 : Gini Index={}\n'.format(replaceUnk_avg_test[2].round(4)))

