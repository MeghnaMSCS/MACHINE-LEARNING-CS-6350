import pandas as pd
from My_SVM import SVM
import numpy as np

data_names = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

trainData = pd.read_csv('/Users/u1368460/Documents/Machine Learning/HW4/bank-note/train.csv', names=data_names)

testData = pd.read_csv('/Users/u1368460/Documents/Machine Learning/HW4/bank-note/test.csv', names=data_names)

hyper_c = 100 / 873
eps = 100
rate = 0.01
D = 0.1
inps = (eps, rate, D, 1)
p_svm = SVM(trainData, hyper_c, inps=inps)
train_p_svm = p_svm.tr_P()
errror = train_p_svm.SVMprediction(testData)
print(errror)

tuning = False
if tuning:
    epochNos_tuning = 100
    rates = 0.1 / (2 ** np.arange(1, 8))
    tryouts = 20
    rate_error = np.zeros((tryouts, len(rates)))

    for i in range(tryouts):
        for indx, rate in enumerate(rates):
            p_svm = SVM(trainData, epochNos_tuning, rate, hyper_c, D)
            train_p_svm = p_svm.tr_P()
            rate_error[i, indx] = train_p_svm.SVMprediction(testData)

    avg_error = np.mean(rate_error, axis=0)
    least_error = min(avg_error)
    least_rate = rates[(avg_error == least_error).argmax()]

tuning2 = False
if tuning2:
    rate = least_rate
    d2 = least_rate * np.array([0.25, 0.5, 1, 2, 4])
    d2_error = np.zeros((trials, len(d2)))
    for i in range(tryouts):
        for indx, d in enumerate(d2):
            p_svm = SVM(trainData, epochNos_tuning, rate, hyper_c, D)
            train_p_svm = p_svm.tr_P()
            d2_error[tryouts, indx] = train_p_svm.SVMprediction(testData)

    avg_error = np.mean(d2_error, axis=0)
    least_error = min(avg_error)
    least_d = d2[(avg_error == least_error).argmax()]

hyper_cs = [100 / 873, 500 / 873, 700 / 873]
tryouts = 1
eps = 100
rate = 0.0125
D = 0.003125
way = 1
tr_errors = np.zeros((tryouts, len(hyper_cs)))
te_errors = np.zeros((tryouts, len(hyper_cs)))
if tryouts == 1:
    weights = []
for c, hyper_c in enumerate(hyper_cs):
    for i in range(tryouts):
        inps = (eps, rate, D, way)
        p_svm = SVM(trainData, hyper_c, inps=inps)
        train_p_svm = p_svm.tr_P()

        errror = train_p_svm.SVMprediction(trainData)
        tr_errors[i, c] = errror
        errror = train_p_svm.SVMprediction(testData)
        te_errors[i, c] = errror
    if tryouts == 1:
        weights.append(train_p_svm.weight)

hyper_cs = [100 / 873, 500 / 873, 700 / 873]
tryouts = 1
eps = 100
rate = 0.0125
D = 0.003125
way = 2
tr_error2 = np.zeros((tryouts, len(hyper_cs)))
te_error2 = np.zeros((tryouts, len(hyper_cs)))

if tryouts == 1:
    weights2 = []
for c, hyper_c in enumerate(hyper_cs):
    for i in range(tryouts):
        inps = (eps, rate, D, way)
        p_svm = SVM(trainData, hyper_c, inps=inps)
        train_p_svm2 = p_svm.tr_P()

        errror = train_p_svm2.SVMprediction(trainData)
        tr_error2[i, c] = errror
        errror = train_p_svm2.SVMprediction(testData)
        te_error2[i, c] = errror
    if tryouts == 1:
        weights2.append(train_p_svm2.weight)

if tryouts == 1:

    output = f"""Weights from Primal SVM  

First schedule - 
Weight_C_1 = {weights[0]}
Weight_C_2 = {weights[1]}
Weight_C_3 = {weights[2]}    

Second schedule - 
Weight_C_1 = {weights2[0]}
Weight_C_2 = {weights2[1]}
Weight_C_3 = {weights2[2]}  

"""
print(output)

# DUAL FORM
hyper_c = 100 / 873
k = 'None'
g = None
inps = (k, g)
d_svm = SVM(trainData, hyper_c, inps=inps, type='D')
apoint = d_svm.solutionD()
weight, bias = d_svm.getparams()
errror = d_svm.dualform(testData)
print(errror)

k = 'None'
g = None
hyper_cs = [100 / 873, 500 / 873, 700 / 873]
d_weight = []
d_bias = []
d_tr_errors = []
d_te_errors = []
for i in hyper_cs:
    inps = (k, g)
    d_svm = SVM(trainData, i, args=inps, type='D')
    apoint = d_svm.solutionD()
    weight, bias = d_svm.getparams()

    d_weight.append(weight)
    d_bias.append(bias)

    errror = d_svm.dualform(trainData)
    d_tr_errors.append(errror)
    errror = d_svm.dualform(testData)
    d_te_errors.append(errror)

# OUTPUT OF DUAL FORM
output = f"""
Weights from Dual SVM

Weights_C_1 = {d_weight[0]}
Weights_C_2 = {d_weight[1]}
Weights_C_3 = {d_weight[2]}

Bias_C_1 = {d_bias[0]}
Bias_C_2 = {d_bias[1]}
Bias_C_3 = {d_bias[2]}

Training_Errors_C1 = {d_tr_errors[0]}
Training_Errors_C2 = {d_tr_errors[1]}
Training_Errors_C3 = {d_tr_errors[2]}

Testing_Errors_C1 = {d_te_errors[0]}
Testing_Errors_C2 = {d_te_errors[1]}
Testing_Errors_C3 = {d_te_errors[2]}

"""

print(output)

# DUAL FORM GAUSSIAN KERNEL
hyper_c = 500 / 873
g = 1
k = 'G'
inps = (k, g)
d_svm_G = SVM(trainData, hyper_c, args=inps, type='D')

weight_g, bias_g = d_svm_G.getparams()
error_k_1 = d_svm_G.dualform(trainData)
error_k_2 = d_svm_G.dualform(testData)
print('Error in Training', error_k_1)
print('Error in Testing', error_k_2)
aguess = apointg.x

hyper_cs = [100 / 873, 500 / 873, 700 / 873]
gs = [0.1, 0.5, 1, 5, 100]
k = 'G'
d_tr_erg = np.zeros((len(hyper_cs), len(gs)))
d_te_erg = np.zeros((len(hyper_cs), len(gs)))
mds = []
j = 1
for r, hyper_c in enumerate(hyper_cs):
    for c, g in enumerate(gs):
        print(f'Dual SVM : gamma value = {g},Hyperparameter C = {hyper_c} ')
        inps(k, g)
        d_svm_G = d_svm_G.dualform(guess=aguess)
        weight_g, bias_g = d_svm_G.getparams()
        mds.append(d_svm_G)

        error_knl = d_svm_G.dualform(trainData)
        d_tr_erg[r, c] = error_knl
        error_knl = d_svm_G.dualform(testData)
        d_te_erg[r, c] = error_knl
        j += 1
        print('Condition complete')

output = f""" 

Errors in Training:
{d_tr_erg}

Error in Testing:
{d_te_erg}


"""

print(output)

n_spt = []
for j in range(15):
    mdl = mds[j]
    jindx1 = mdl.apoint > 1e-6
    jindx2 = mdl.apoint < mdl.hyper_c - 1e-6
    jindx = np.logical_and(jindx1, jindx2)
    n_spt.append(np.sum(jindx))

eqs = []
for j in range(5, 9):
    k = j + 1
    als1 = mds[j].apoint
    als2 = mds[k].apoint
    eq = np.sum(als1 == als2)
    eqs.append(eq)



