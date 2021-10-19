import numpy as np

class methodLMS:
    def __init__(self, inputt, T, l_r, vbs=False):
        self.a = np.append(np.ones((len(inputt), 1)), inputt.iloc[:, :-1], 1)
        self.b = inputt.iloc[:, -1]
        self.T = T
        self.l_r = l_r
        self.inn_wt = np.zeros((len(self.a.T),))
        self.conv = False
        self.J = np.zeros((T,))
        self.vbs = vbs


    def Batch(self,wt):
        for tt in range(self.T):
            self.J[tt]=0.5*sum((self.b-wt.dot(self.a.T))**2)
            gradient_j=(-(self.b-wt.dot(self.a.T))).dot(self.a)
            latest_wt = wt-self.l_r*gradient_j
            n=self.testConv(wt,latest_wt,tt)
            if self.conv:
                print(f'Linear regression={self.l_r},complete, was converged to={self.conv}\n')
                return wt
            wt = latest_wt
        print(f'Linear regression={self.l_r},complete, was converged to={self.conv}\n')
        print(f'Weight={n}\n')
        return wt

    def Stoch(self,wt,vbs=False):
        for tt in range(self.T):
            self.J[tt]=0.5*sum((self.b-wt.dot(self.a.T))**2)
            ii = np.random.choice(np.arange(len(self.b)))
            gradient_J= -(self.b[ii]-sum(wt*self.a[ii,:])) * self.a[ii,:]
            latest_wt=wt-self.l_r*gradient_J
            n=self.testConv(wt,latest_wt,tt)
            if self.conv:
                print(f'Linear regression={self.l_r},complete, was converged to={self.conv}\n')
                return wt
            wt = latest_wt
            if vbs:
                print('Weight=',wt,'Gradient=',gradient_J)
        print(f'Linear regression={self.l_r},complete, was converged to={self.conv}\n')
        print(f'Weight={n}\n')
        return wt


    def testConv(self,wt,latest_wt,tm):
        n=np.linalg.norm(wt-latest_wt)
        if n<1e-6 and tm!=0:
            self.conv=True
        return n

def callBatchLMS(self):
    inn_wt=self.inn_wt
    last_w = self.Batch(inn_wt)
    return last_w

def callStochLMS(self):  
    inn_wt=self.inn_wt
    last_w = self.Stoch(inn_wt,vbs=self.vbs)
    return last_w
