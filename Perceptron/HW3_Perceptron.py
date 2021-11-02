import numpy as np
import pandas as pd

class perceptronAlgo:
    def __init__(self,T,rate,input,type='std'):
        self.T=T
        self.rate=rate
        self.training=np.append(np.ones((len(input),1)),input.iloc[:,:-1],1)
        self.lb = (np.array(input.iloc[:,-1])*2)-1
        self.wv=np.zeros((self.training.shape[1],))
        if type=='vote':
            self.w_v=[]
            self.ex=[]
        elif type=='average':
            self.avg = np.zeros((self.training.shape[1],))
        self.type=type

    def shuffling(self):
        indx=np.random.choice(np.arange(len(self.training)),len(self.training),replace=False)
        inputShuffle = self.training[indx,:]
        labelShuffle = self.lb[indx]
        return inputShuffle, labelShuffle

    def trainpAlgo(self):
        for i in range(self.T):
            input, lb=self.shuffling()
            for x in range(len(input)):
                error_p = lb[x]*self.wv.dot(input[x,:].T)
                if error_p<=0:
                    self.wv+=self.rate*lb[x]*input[x,:]
                else:
                    continue
        return self

    def trainVoteAlgo(self):
        ch = 1
        for i in range(self.T):
            input, lb=self.shuffling()
            for x in range(len(input)):
                error_p=lb[x]*self.wv.dot(input[x,:].T)
                if error_p<=0:
                    self.wv+=self.rate*lb[x]*input[x,:]
                    self.w_v.append(self.wv.copy())
                    self.ex.append(ch)
                    ch=1
                else:
                    ch+=1
        return self

    def trainAvgAlgo(self):
        for i in range(self.T):
            input, lb=self.shuffling()
            for x in range(len(input)):
                error_p = lb[x]*self.wv.dot(input[x,:].T)
                if error_p<=0:
                    self.wv+=self.rate*lb[x]*input[x,:]
                else:
                    pass
                self.avg+=self.wv
        return self

    def Pprediction(self,testing):
        testingData = np.append(np.ones((len(testing),1)),testing.iloc[:,:-1],1)
        testinglb = np.array(testing.iloc[:,-1])>0
        testinglb=(testinglb*2)-1
        if self.type=='std':
            makePredict = self.wv.dot(testingData.T)>=0
            makePredict = (makePredict*2)-1
        elif self.type=='vote':
            makePredict=np.zeros_like(testinglb)
            for k in range(len(self.ex)):
                iti = self.w_v[k].dot(testingData.T)>=0
                makePredict+=self.ex[k]*((iti*2)-1)
            makePredict=((makePredict>=0)*2)-1
        else:
            makePredict=self.avg.dot(testingData.T)>=0
            makePredict=(makePredict*2)-1


        error_diff = makePredict!=testinglb
        errror = sum(error_diff) / len(error_diff)
        return errror


