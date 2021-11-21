import pandas as pd
import numpy as np
import HW4_DUAL

trainData = pd.read_csv('/Users/u1368460/Documents/Machine Learning/HW4/bank-note/train.csv', header=None)

data_val = trainData.values
nos_c = data_val.shape[1]
nos_r = data_val.shape[0]
data_1 = np.copy(data_val)
data_1[:,nos_c-1] = 1
data_2 = data_val[:,nos_c-1]
data_2 = 2 * data_2-1

testData = pd.read_csv('/Users/u1368460/Documents/Machine Learning/HW4/bank-note/test.csv', header=None)
data_val = testData.values
nos_c = data_val.shape[1]
nos_r = data_val.shape[0]
data_3 = np.copy(data_val)
data_3[:, nos_c-1] = 1
data_4 = data_val[:, nos_c-1]
data_4 = 2*data_4-1

hyper_cs = np.array([100,500,700])
hyper_cs = hyper_cs / 873
gs = np.array([0.1,0.5,1,5,100])
SVM_f = HW4_DUAL.SVM()
for hyper_c in hyper_cs:
    print('Hyper parameter C =',hyper_c )
    SVM_f.get_hyper_c(hyper_c)

    #Dual Form
    wght = SVM_f.tr_D(data_1[:, [x for x in range(nos_c-1)]], data_2)
    wght = np.reshape(wght,(5,1))

    pdiction = np.matmul(data_1, wght)
    pdiction[pdiction>0] = 1
    pdiction[pdiction<=0] = -1
    tr_error = np.sum(np.abs(pdiction-np.reshape(data_2, (-1,1))))/2/data_2.shape[0]

    pdiction = np.matmul(data_3, wght)
    pdiction[pdiction>0] = 1
    pdiction[pdiction<=0] = -1

    te_error = np.sum(np.abs(pdiction-np.reshape(data_4,(-1,1))))/2/data_4.shape[0]
    print('Dual SVM: Training Error',tr_error,'Testing Error', te_error)

    #Gaussian
    hc=0
    for g in gs:
        print('Gamma values=',g)
        SVM_f.get_g(g)
        alp = SVM_f.tr_gk(data_1[:,[x for x in range(nos_c-1)]], data_2)
        indx = np.where(alp>0)[0]
        print('Support vector no=',len(indx))

        y=SVM_f.p_gk(alp,data_1[:,[x for x in range(nos_c-1)]],data_2,
                     data_1[:,[x for x in range(nos_c-1) ]])
        tr_error = np.sum(np.abs(y-np.reshape(data_2,(-1,1))))/2/data_2.shape[0]

        y = SVM_f.p_gk(alp,data_1[:,[x for x in range(nos_c-1)]], data_2,
                       data_1[:,[x for x in range(nos_c-1)]])
        te_error = np.sum(np.abs(y-np.reshape(data_4,(-1,1))))/2/data_4.shape[0]
        print('Non Linear SVM Training Error=', tr_error,'Testing Error=',te_error)

        if hc>0:
            overlap = len(np.intersect1d(indx, old_indx))
            print('Overlap=', overlap)
        hc = hc+1
        old_indx = indx












