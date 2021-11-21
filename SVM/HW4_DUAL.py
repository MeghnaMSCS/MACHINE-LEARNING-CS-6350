import numpy as np
import scipy.optimize as opt

class SVM:
    def __init__(self):
        self.hyper_c = 10
        self.rate = 0.1
        self.D = 0.1
        self.eps = 100
        self.g = 0.1

    def get_hyper_c(self, hyper_c):
        self.hyper_c = hyper_c

    def get_rate(self, rate):
        self.rate = rate

    def get_D(self,D):
        self.D = D

    def get_eps(self,eps):
        self.eps = eps

    def get_g(self,g):
        self.g = g

    def objct(selfself, alp, x, y ):
        l = 0
        l = l-np.sum(alp)
        axy = np.multiply(np.multiply(np.reshape(alp,(-1,1)), np.reshape(y,(-1,1))),x)
        l=l+0.5*np.sum(np.matmul(axy, np.transpose(axy)))
        return l

    def cnstraint(self,alp,y):
        t = np.matmul(np.reshape(alp,(1,-1)), np.reshape(y, (-1,1)))
        return t[0]

    def tr_D(self,x,y):
        nos_smp = x.shape[0]
        bds = [(0,self.hyper_c)]*nos_smp
        cnstraint = ({'type': 'eq', 'fun' : lambda alp: self.cnstraint(alp,y)})
        alp0 = np.zeros(nos_smp)
        rstore = opt.minimize(lambda alp: self.objct(alp,x,y), alp0, method='SLSQP',bounds=bds,
                              constraints=cnstraint, options={'disp':False})

        wght = np.sum(np.multiply(np.multiply(np.reshape(rstore.x, (-1,1)), np.reshape(y,(-1,1))),x),axis=0)
        indx = np.where((rstore.x>0)&(rstore.x < self.hyper_c))
        b = np.mean(y[indx]-np.matmul(x[indx,:],np.reshape(wght,(-1,1))))
        wght = wght.tolist()
        wght.append(b)
        wght = np.array(wght)
        return wght

    def gkernel(self,x1,x2,g):
        mat1 = np.tile(x1,(1,x2.shape[0]))
        mat1 = np.reshape(mat1,(-1,x1.shape[1]))
        mat2 = np.tile(x2,(x1.shape[0],1))
        kk = np.exp(np.sum(np.square(mat1-mat2),axis=1)/-g)
        kk = np.reshape(kk,(x1.shape[0],x2.shape[0]))
        return kk

    def objctgk(self, alp, kk,y):
        l=0
        l=l-np.sum(alp)
        aly=np.multiply(np.reshape(alp, (-1,1)),np.reshape(y,(-1,1)))
        alyaly = np.matmul(aly, np.transpose(aly))
        l = l+0.5*np.sum(np.multiply(alyaly,kk))
        return l

    def tr_gk(self,x,y):
        nos_smp = x.shape[0]
        bds = [(0,self.hyper_c)] * nos_smp
        cnstraint = ({'type':'eq', 'fun':lambda alp:self.cnstraint(alp,y)})
        alp0 = np.zeros(nos_smp)
        kk = self.gkernel(x,x,self.g)
        rstore = opt.minimize(lambda alp: self.objctgk(alp,kk,y), alp0, method='SLSQP',
                              bounds=bds, constraints=cnstraint, options={'disp':False})
        return rstore.x

    def p_gk(self, alp, x0, y0, x):
        kk = self.gkernel(x0, x, self.g)
        kk = np.multiply(np.reshape(y0,(-1,1)),kk)
        y = np.sum(np.multiply(np.reshape(alp, (-1,1)),kk), axis=0)
        y= np.reshape(y,(-1,1))
        y[y>0] = 1
        y[y <= 0] = -1
        return y
