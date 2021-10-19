import numpy as np
from HW1_ID3Algorithm import DT, ID3_algo

class boost:
    def __init__(self,data,tree,depth=1,k=None):
        self.data = data
        self.labels=np.array(train.iloc[:,-1])
        self.DTinit=np.ones((len(data),))/len(data)
        self.tree=tree
        self.depth=depth
        self.l_i = []
        self.a=np.zeros((tree,))
        self.e=np.zeros((tree,))
        self.ew=np.zeros((tree,))
        self.k=k


    def vote(self,stumps,tree,d,num=0):
        init_error= testtree(self.d,init_stump,multiples=d,num=num)
        H_T,error_total=testID3(init_error)

        if error_total>0.5:
            print(f'The total error is equal to {error_total},that is higher than 50%')
        self.ew[tree]=error_total
        self.errors[tree]=1-sum(H_T)/len(H_T)
        self.a[tree]=0.5*np.log((1-error_total)/(error_total))

        return H_T

    def status(self,tree):
        pt = np.round(100*tree/ self.tree)
        if len(self.l_i)!=self.tree:
            print(f'{pt}% over. {tree} TREES Created')
        else:
            print(f'{pt}% over. {tree} TREES made')


    def mappings(selfself,H_T):
        return H_T*2-1

    def state_multiples(self,d,tree,H_T):
        qw=self.mappings(H_T)
        d_pp=d*np.exp(-self.a[tree]*qw)
        aa=np.sum(d_pp)
        d_pp/=aa
        return d_pp

    def lloop(self,d):
        print('Begin to train')
        for tree in range(self.tree):
            if (tree) % np.round(self.tree/10)==0:
                self.status(tree)
            stumps=DT(self.data,num=1,depth=self.depth,multiples=d)
            ID3_algo(stumps)
            self.l_i.append(stumps)
            H_T=self.vote(stumps,tree,d,num=1)
            dpp=self.state_multiples(d,tree,H_T)
            d=dpp
        print('Finished to train')

    def trainBoost(self,data):
        prediction=[]
        for tree in range(self.tree):
            if (tree) % np.round(self.tree/10)==0:
                self.status(tree)
            t_i=self.l_i[tree]

            train_i=testtree(data,t_i,multiples=t_i.multiples,num=1)
            testinit=testtree(data,t_i,multiples=t_i.multiples,num=1)
            testID3(testinit)
            prediction.append(testinit.prediction)
        print('Finished testing')
        return prediction

def BoostAlgo(self):
    initi=self.init.copy()
    self.lloop(initi)


def tBoost(self,data):
    H_T=np.array(self.trainBoost(data))
    H_T=(np.vectorize(self.key.get)(H_T)).tree
    a=np.array(self.a)
    a_h = a * H_T
    er=np.zeros((self.tree,))
    t_l = data.iloc[:,-1]
    t_l=np.array([self.key[t_l[i]]for i in range(len(data))])
    for tree in range(self.T):
        h = np.sum(alpha_h[:,:tree+1],axis=1)>0
        h=h*2-1
        er[tree]=sum(h!=t_l) / len(t_l)
    return er