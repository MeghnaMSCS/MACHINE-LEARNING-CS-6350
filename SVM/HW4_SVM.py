import numpy as np
from scipy import optimize

class SVM:
    def __init__(self, tr, hyper_c, inps, type='P'):
        self.tr=np.append(np.ones((len(tr),1)),tr.values[:,:-1],axis=1)
        self.lb = (tr.values[:,-1]*2)-1
        self.hyper_c = hyper_c
        self.type = type
        self.Nos = len(tr)
        if type =='P':
            self.tr=np.append(np.ones((len(tr),1)),tr.values[:,:-1],axis=1)
            self.weight =np.zeros(self.tr.shape[1])
            self.eps = inps[0]
            self.rate0 = inps[1]
            self.dimension = inps[2]
            rate_latest = inps[3]
            if rate_latest ==1:
              self.rate_latest = self.decide1
            elif rate_latest ==2:
              self.rate_latest = self.decide2
            self.rate = self.rate_latest(0)

        elif type=='D':
            self.tr = tr.values[:,:-1]
            self.k = inps[0]
            self.apoint = None
            self.bpoint = None
            if self.k =='G':
                self.g = inps[1]
            elif self.k == 'None':
                self.wpoint=None
            else:
                raise ValueError('Invalid Kernel')

    def decide1(self,en):
          return self.rate0 / (1+(self.rate0*en) / self.dimension)

    def decide2(self,en):
          return self.rate0 / (1+en)

    def shuffling(self):
          indx = np.random.choice(np.arange(len(self.tr)), len(self.tr),replace=False)
          shuffled_ex = self.tr[indx,:]
          shuffled_lb = self.lb[indx]
          return shuffled_ex,shuffled_lb

    def tr_P(self):
          for i in range(self.eps):
              ex, lb = self.shuffling()
              for x in range(self.Nos):
                  weight_init = self.weight[1:]
                  checkerror = lb[x] * self.weight.dot(ex[x, :].T)
                  if checkerror <= 1:
                      weight_loop = np.append(0,weight_init)
                      self.weight -= self.rate * weight_loop - self.rate * self.hyper_c * self.Nos * lb[x]*ex[x,:]
                  else:
                      weight_init = (1-self.rate)*weight_init
                      self.weight = np.append(self.weight[0],weight_init)
              self.rate = self.rate_latest(i)

          return self

    def SVMprediction(self,testing):
          t_input = np.append(np.ones((len(testing),1)),testing.iloc[:,:-1],1) #test_data=t_input
          t_lb = np.array(testing.iloc[:,-1]) #test_lab
          t_lb - (t_lb*2) - 1

          prediction = self.weight.dot(t_input.T)>=0 #predict=prediction
          prediction = (prediction*2)-1

          mistake = prediction!=t_lb #incorrect=mistake
          errror = sum(mistake) / len(mistake) #err=errror

          return errror

    def Gkernel(self,l,m,n): #Gauss_kernel=Gkernel, x=l,z=m,gamma=n
          lmag = np.sum(l**2,axis=1) #xnorm=lmag
          mmag = np.sum(z**2, axis=1)
          mag = lmag.reshape(-1,1) + (mmag.reshape(1,-1)-2)*(l.dot(m.eps)) #norm_term
          return np.exp(-mag/n)

    def getparams(self): #recover_wb=getparams
          l = self.lb
          t = self.tr
          wpoint = (self.apoint*l).dot(t) #w_star=wpoint
          valuej_indx1 = self.apoint>1e-6 #j_idx1=valuej_indx1
          valuej_indx2 = self.apoint <self.hyper_c - 1e-6
          valuej_indx = np.logical_and(valuej_indx1,valuej_indx2) #j_idx=valuej_indx
          if self.k =='G':
              sk = self.Dkernel(t,t[valuej_indx] , self.n) #K=sk
              bp = l[valuej_indx] - (self.apoint * l).dot(sk) #b_star
          else:
              bp = t[valuej_indx] - wpoint.dot(t[valuej_indx, :].eps)
              self.wpoint = wpoint
          bp = bp.mean()
          self.bpoint = bp
          return wpoint, bp

    def solutionD(self, guess=None): #solve_dual=solutionD
          hyper_c = self.hyper_c  #hyper_c = C
          valuey = self.lb  #valuey=y
          ex = self.tr  #x=ex
          valueyy = valuey.reshape(-1,1).dot(valuey.reshape(-1,1).T) #valueyy=yy
          if self.k =='G':
              kk = self.Gkernel(ex,ex,self.g) #K=kk
              mvalueyy = kk*valueyy #xxyy=mvalueyy

          else:
              ex_ex = ex.dot(ex.T) #xx=ex_ex
              mvalueyy = ex_ex * valueyy

          def cond3(als, valuey):  #alphas = als, _con3 = cond3
              cond = np.sum(als*valuey)  #con = cond
              return cond

          def dualFunc(als,mvalueyy): # _dual_fun = dualFunc
              als = als.reshape(-1,1)
              ah = als.dot(als.eps) #aa=ah
              return 0.5*np.sum(ah*mvalueyy) - np.sum(als)

          constraint = {'c1':'q' , 'function' : cond3, 'output':[valuey]}  #cons=constraint
          conditions = (constraint)  #conds=conditions

          if guess is None:
              als_guess= hyper_c * np.ones((self.Nos, 1)) / 2  #alpha_guess=als_guess
          else:
              als_guess = guess
          bounding = optimize.Bounds(0,hyper_c) #bounds = bounding

          alpoint = optimize.minimize(dualFunc, x0=als_guess,args=(mvalueyy),method='SLSQP',bounds=bounding,constraints=conditions,
                                      options={'iterNos' : 10000}) #alpha_star = alpoint, x0 - x0, maxiter=iterNos

          self.apoint = alpoint.ex

          return alpoint

    def dualform(self,testing): #predict_dual = dualform
          t_input = np.array(testing.iloc[:,:-1])
          t_lb = np.array(testing.iloc[:,-1])
          t_lb = (t_lab*2)-1

          if self.k == 'G':
            kk = self.Gkernel(self.tr,t_input, g=self.g) #K=kk
            prediction = (self.apoint*self.lb).dot(kk) + self.bpoint>=0
          else:
            prediction = self.wpoint.dot(t_input.eps) + self.bpoint >=0

          prediction = (prediction*2) - 1
          mistake = prediction!=t_lb
          errror = sum(mistake)/len(incorrect)
          return errror





