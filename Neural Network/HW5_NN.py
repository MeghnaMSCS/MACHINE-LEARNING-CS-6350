import numpy as np

class nwlayer:
    def __init__(self, nm, inp, out, nnlayer = 'hid' , iniw = 'gs'):
        self.nnlayer=nnlayer
        self.nwlayer_nm = nm
        self.nd = np.zeros(out)
        if self.nnlayer =='hid':
            self.n = np.zeros(out)
        else:
            pass

        if iniw =='gs':
            np.random.seed(15)
            self.wghs = np.random.randn(inp+1,out)
        elif iniw =='z':
            self.wghs = np.zeros((inp+1,out))
        else:
            self.wghs=iniw

        self.dn = np.zeros_like(self.nd)
        self.wghs_d = np.zeros_like(self.wghs)

class nnetwork:
    def __init__(self, wt,dt,epochNos=5,rate0=0.01,f0=0.005,f=3,iniw='gs'):
        self.wt=wt
        self.f=f
        self.epochNos=epochNos
        self.tr=dt.iloc[:,:-1].values
        self.lb = ((dt.iloc[:,-1].values)*2)-1
        self.nos = len(self.tr)
        self.rate0 = rate0
        self.f0 = f0

        if (iniw=='gs') or (iniw=='z'):
            input_ly = [nwlayer(1,self.tr.shape[1],wt,nnlayer='hid',iniw=iniw)]
            hid_ly = [nwlayer(j,wt,wt,nnlayer='hid',iniw=iniw) for j in range(2,f,1)]
            out_ly = [nwlayer(f,wt,1,nnlayer='outpt',iniw=iniw)]
            self.lys = input_ly+hid_ly+out_ly
        else:
            hid_ly = [nwlayer(j,wt,wt,nnlayer='hid',iniw=iniw[j-1]) for j in range(1,f,1)]
            out_ly = [nwlayer(f,wt,1,nnlayer='outpt',iniw=iniw[-1])]
            self.lys = hid_ly + out_ly

    def fwd(self,fts):
        for ly in self.lys:
            ns = ly.nwlayer_nm
            if len(fts.shape) == 1:
                fts = fts.reshape(1,-1)

            if ns==1:
                bb = np.ones((fts.shape[0],1))
                inp = np.append(bb,fts,axis=1)
            else:
                bb=np.ones((self.lys[ns-2].nd.shape[0],1))
                inp = np.append(bb,self.lys[ns-2].nd, axis=1)

            nd_val = inp.dot(ly.wghs)

            if ly.nnlayer =='outpt':
                ly.nd = nd_val
            else:
                ly.n=nd_val
                ly.nd = funcsigm(nd_val)

        return self.lys[2].nd

    def backpro(self,fts,lb):
        rev_lys = np.flip(self.lys.copy())
        for ly in rev_lys:
            ns = ly.nwlayer_nm
            dn_ly = self.f-(ns-1)
            if ly.nnlayer=='outpt':
                dn_nd = np.append(1,rev_lys[dn_ly].nd)

                fL = fLoss(ly.nd,lb)
                ly.dn = fL
                ly.wghs_d = fLin(fL, dn_nd)

            elif ly.nnlayer == 'hid':
                if ly.nwlayer_nm==1:
                    dn_nd = np.append(1,fts)
                else:
                    dn_nd = np.append(1,rev_lys[dn_ly].nd)

                u_ly = self.f-(ns+1)
                if u_ly ==0:
                    f_u = rev_lys[u_ly].dn
                    wghs_u = rev_lys[u_ly].wghs
                    f_sig = fSig(ly.n)
                    f_N = f_u.reshape(1,-1).dot(wghs_u.T)
                else:
                    f_u = rev_lys[u_ly].dn[:,1:]
                    wghs_u = rev_lys[u_ly].wghs
                    f_sig = fSig(ly.n)
                    f_sig_u = fSig(rev_lys[u_ly].n)
                    f_N = (f_u*f_sig_u).reshape(1,-1).dot(wghs_u.T)

                ly.dn =  f_N
                fN = f_N[:,1:]
                ly.wghs_d = dn_nd.reshape(-1,1).dot((fN*f_sig).reshape(1,-1))

    def tr_nw(self):
        for eN in range(self.epochNos):
            rate = sched(self.rate0,self.f0,eN)
            dt = self.tr
            lb=self.lb
            for x in range(1):
                fts=dt[x,:]
                lbb =lb[x]
                print(fts,lbb)
                self.fwd(fts)
                self.backpro(fts,lbb)
                for ly in self.lys:
                    if ly.nwlayer_nm==self.f:
                        ly.wghs -= rate*ly.wghs_d.T
                    else:
                        ly.wghs -= rate * ly.wghs_d

        return self

    def use_nw(self, testdata):
        tdata = testdata.iloc[:,:-1].values
        tlb = testdata.iloc[:,-1].values.reshape(-1,1)
        tlb = (tlb*2)-1

        pdict = ((self.fwd(tdata)>=0)*2)-1

        diff = pdict!=tlb
        error = sum(diff)/len(diff)

        return error


    def use_tr(self,tr,te):
        tr_error = np.zeros(self.epochNos)
        te_error = np.zeros(self.epochNos)
        for eN in range(self.epochNos):
            rate=sched(self.rate0,self.f0,eN)
            dt, lb = shuffling(self.tr,self.lb)
            for x in range(self.nos):
                fts = dt[x,:]
                lbb = lb[x]
                self.fwd(fts)
                self.backpro(fts,lbb)
                for ly in self.lys:
                    if ly.nnlayer == 'outpt':
                        ly.wghs -= rate*ly.wghs_d.T
                    else:
                        ly.wghs -= rate * ly.wghs_d

            error = self.use_nw(tr)
            tr_error[eN] = error
            error = self.use_nw(te)
            te_error[eN] = error

        return self, tr_error, te_error

def funcsigm(ip):
    return 1/(1+np.exp(-ip))

def lss(ip1,ip2):
    return 0.5*(ip1-ip2) **2

def fLoss(ip1,ip2):
    return ip1-ip2

def fLin(ip1,ip2):
    return ip1*ip2

def fSig(ip):
    return funcsigm(ip) * (1-funcsigm(ip))

def shuffling(ip1,ip2):
    indx = np.random.choice(np.arange(len(ip1)),len(ip1), replace=False)

    data_shuffle = ip1[indx,:]
    lb_shuffle = ip2[indx]
    return data_shuffle, lb_shuffle

def sched(rate0, f0, eN):
    return rate0/(1+(rate0*eN)/f0)

def lin(ip1,ip2):
    return np.sum(ip1*ip2)

def nodes(ip1,ip2,ver=False, de=None):
    summing = lin(ip1,ip2)
    if ver:
        outputt = f"""  {de}
        List of features = {ip1}
        List of weights = {ip2}
        Linear units = {summing}
        Sigmoid function = {funcsigm(summing)} 

        """
        print(outputt)
    return funcsigm(summing)

