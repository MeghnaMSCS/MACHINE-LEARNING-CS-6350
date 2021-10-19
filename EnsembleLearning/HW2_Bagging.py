import numpy as np
from HW1_ID3Algorithm import DT, ID3_algo, testtree,testID3

class baggingAlgo:

    def __init__(self,M,tree,data,num=0,k=None,rF=0,gs=6,vbs=1,glo_o=0,glo_df=None):
        self.M=M
        self.tree=tree
        self.data=data
        self.errors=np.zeros((tree,))
        self.a=np.zeros((tree,))
        self.ti=[]
        self.num=num
        self.k=k
        if M<0.5*len(data):
            self.tinys=True
            self.glo_df=data
        else:
            self.tinys=False
        self.rF=rF
        self.gs=gs
        self.vbs=vbs
        if glo_o:
            self.tinys=True
            self.glo_df=glo_df

    def different_draw(self):
        indx = np.random.choice(np.arange(len(self.data)),self.M)
        return self.data.iloc[list(indx)]

    def Update_in_bag(self,tree):
        pt = np.round(100*tree/self.tree)
        if self.vbs:
            if len(self.ti)!=self.tree:
                print(f'{pt}% over.{tree} trees done')
            else:
                print(f'{pt}%over. {tree} tested')

    def vote2(self,ti,tree,num=0):
        error_i = testtree(self.data,initTree,num=num)
        H_T, errorTotal = ID3_algo(error_i)
        self.errors[tree]=errorTotal
        self.a[tree]=0.5*np.log((1-errorTotal)/errorTotal)

    def llopp_bag(self):
        for tree in range(self.tree):
            if (tree)%np.round(self.tree/10)==0:
                self.Update_in_bag(tree)
            bstrap=self.different_draw()
            if self.tinys:
                initTree=DT(bstrap, num=self.num, tinys=self.tinys, glo_df=self.glo_df, rF=self.rF, gs=self.gs)
            else:
                initTree=DT(bstrap,num=self.num,rF=self.rF,gs=self.gs)
            self.ti.append(initTree)
            ID3_algo(initTree)
            self.vote2(initTree,tree,num=self.num)
        if self.vbs:
            print('Completed 100%')

    def mapping2(self,H,k):
        H_map = [k[i] for i in H]
        return np.array(H_map)

    def test_lloop_bag(selfself,data):
        prediction=[]
        for tree in range(self.tree):
            if (tree)%np.round(self.tree/10)==0:
                self.Update_in_bag(tree)
            testInit = testtree(data,self.ti[tree],num=self.num)
            testID3(testInit)
            prediction.append(testInit.prediction)
        if self.vbs:
            print('Completed 100%')
        return prediction

def call_bagging(self):
    self.llopp_bag()

def test_bagging(self,data):
    H_T=np.array(self.test_lloop_bag(data))
    H_T=(np.vectorize(self.k.get)(H_T)).tree
    a=np.array(self.a)
    a_h = a*H_T
    errror = np.zeros((self.tree,))
    t_l = data.iloc[:,-1]
    t_l=np.array([self.k[t_l[i]]for i in range(len(data))])
    for t in range(self.tree):
        h=np.sum(a_h[:,:tree+1],axis=1)>0
        h=h*2-1
        errror[tree]=sum(h!=t_l) / len(t_l)
    return errror

