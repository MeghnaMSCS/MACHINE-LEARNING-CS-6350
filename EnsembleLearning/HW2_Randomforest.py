import numpy as np
import pandas as pd
from HW2_Bagging import baggingAlgo, call_bagging
from HW1_ID3Algorithm import testtree, testID3
import glob

def testReplace(M,train):
    indx=np.random.choice(np.arange(len(train)),M,replace=0)
    return train.iloc[list(indx)]

def callBagTree(M,treeBag,train,k,glo_df,gs):
    Initbag=baggingAlgo(M,treeBag,train,num=0,k=k,glo_o=1,glo_df=glo_df,vbs=0,rF=1,gs=gs)
    call_bagging(Initbag)
    return Initbag

def makePrediction(Initbag, check,k):
    bag1 = np.array(Initbag.test_lloop_bag(check))
    testanytree=testtree(check,Initbag.ti[0],num=1)
    testID3(testanytree)
    atree = np.array(testanytree.prediction)

    abag = (np.vectorize(k.get)(abag)).tree
    a=np.array(Initbag.a)
    a_h = a*abag
    h = np.sum(a_h,axis=1)>0
    abag = h*2-1

    atree = np.vectorize(key.get)(atree)
    return abag,atree

def lgr(abag,atree,anum):
    if anum==0:
       ncsv = int(len(glob.glob('*.csv'))/2)
       abags=pd.DataFrame(abag,columns=[anum])
       atrees=pd.DataFrame(atree,columns=[anum])
       abags.to_csv('randomForest'+ str(ncsv) + '.csv',index=False)
       atrees.to_csv('randomTrees' + str(ncsv) + '.csv', index=0)
    else:
        ncsv = int(len(glob.glob('*.csv'))/2-1)
        titleBag='randomForest'+str(ncsv) + '.csv'
        titletree='randomTrees'+str(ncsv) + '.csv'
        presentbag=pd.read_csv(titleBag)
        presenttree=pd.read_csv(titletree)
        presentbag[anum] = abag
        presenttree[anum]=atree
        presentbag.to_csv(titleBag,index=0)
        presenttree.to_csv(titletree,index=0)

#Testing on Bank problem

nametitles=['age','job','marital','education','default','balance','housing','loan','contact','day','month',
          'duration','campaign','pdays','previous','poutcome','y']

trainData = pd.read_csv('/Users/u1368460/Documents/Machine Learning/ASSIGNMENTS/ASSIGNMENT 2/bank/train.csv', names=nametitles)
testData=pd.read_csv('/Users/u1368460/Documents/Machine Learning/ASSIGNMENTS/ASSIGNMENT 2/bank/test.csv', names=nametitles)

q=1000
bag = 50
k= {'NO': 0 , 'YES': 1}
i = 10
magnitude = 5

if i<100:
    print('With less number of iterations')
tme = []
beginn=""" Beginning test. Might take lot of time"""
print(beginn)

for j in range(i):
    print('Building tree')
    trainagain = testReplace(q,trainData)
    bags=callBagTree(q,bag,trainagain,k,glo_df=trainData,gs=magnitude)
    print('Predicting..')
    bag1,tree1 = makePrediction(bags,testData,k)
    lgr(bag1,tree1,j)
