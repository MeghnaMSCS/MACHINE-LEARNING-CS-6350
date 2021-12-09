from HW5_NN import nnetwork
import numpy as np
import pandas as pd
import matplotlib.pyplot as plots

data_names = ['variance','skewness','curtosis','entropy','label']

trainData = pd.read_csv('/Users/u1368460/Documents/Machine Learning/HW5/bank-note 2/train.csv', names=data_names)

testData=pd.read_csv('/Users/u1368460/Documents/Machine Learning/HW5/bank-note 2/test.csv', names=data_names)

epochNos = 50
wght = 25
rate0s = [0.1,0.05,0.01,0.005,0.001]
f0s = [1,0.5,0.1,0.05,0.01]
print('Hyperparameter - Tuning ..Start..')
for rate0 in rate0s:
    for f0 in f0s:
        nnw = nnetwork(wght,trainData,epochNos=epochNos, rate0=rate0,f0=f0,iniw='gs')
        tr_nw,tr_error,te_error = nnw.use_tr(trainData,testData)
        print(f'Learning Rate={rate0}, d= {f0}, Training Error={tr_error[-1]},Testing Error = {te_error[-1]}')
        plots.figure()
        vs = np.arange(epochNos)
        plots.plot(vs,tr_error)
        plots.plot(vs,te_error)

print('Complete\n')
print('Combination is Learning Rate=0.01, d=0.5\n')

epochNos = 50
rate0 = 0.01
f0 = 0.5
wd_nos = [5,10,25,50,100]
tr_werros = []
te_werrors = []
print('Neural Network under different widths')

for wd in wd_nos:
    print(f'Currently running with width = {wd}')
    nnw = nnetwork(wd,trainData,epochNos=epochNos,rate0=rate0,f0=f0,iniw='gs')
    tr_nw,tr_error,te_error = nnw.use_tr(trainData,testData)
    tr_werros.append(tr_error)
    te_werrors.append(te_error)
    print('Complete\n')

figure, axis = plots.subplots(figsize=(8,4))
vs = np.arange(epochNos)
col = ['Green','Orange','Blue','Red','Black']
for j in range(len(tr_werros)):
    plots.plot(vs,tr_werros[j],linewidth=3, color=col[j])
    plots.plot(vs,te_werrors[j],linewidth=3,linestyle='--',color=col[j])
plots.xlim([0,50])
plots.ylim([0,0.1])
axis.tick_params(labelsize=14, width=2, size=7)
for sp in axis.spines:
    axis.spines[sp].set_linewidth(2)
plots.xlabel('EPOCH',fontsize=16)
plots.ylabel('ERROR', fontsize=16)
lists = ['Training w=5','Testing w=5','Training w=10','Testing w=10','Training w=25','Testing w=25',
         'Training w=50','Testing w=50', 'Training w=100','Testing w=100']
plots.legend(lists,fontsize=13)
plots.savefig('Gaussian Initialization',dpi=150, bbox_inches='tight')


epochNos=50
rate0=0.01
f0 = 0.5
wd_nos = [5,10,25,50,100]
tr0_werrors = []
te0_werrors = []
print('Initialized to Zero\n')
print('Executing Neural Network with different widths')
for wd in wd_nos:
    print(f'Currently running with width={wd}')
    nnw = nnetwork(wd,trainData,epochNos=epochNos,rate0=rate0,f0=f0,iniw='zero')
    tr_nw,tr_error,te_error=nnw.use_tr(trainData,testData)
    tr0_werrors.append(tr_error)
    te0_werrors.append(te_error)
    print('Complete\n')

figure,axis = plots.subplots(figsize=(8,4))
vs=np.arange(epochNos)
col = ['Green','Orange','Blue','Red','Black']
for j in range(len(tr_werros)):
    plots.plot(vs,tr0_werrors[j],linewidth=3,color=col[i])
    plots.plot(vs,te0_werrors[j],linewidth=3,linestyle='--',color=col[j])
plots.xlim([0.50])
plots.ylim([0,0.6])
axis.tick_params(labelsize=14,width=2,size=7)
for sp in axis.spines:
    axis.spines[sp].set_linewidth(2)
plots.xlabel('EPOCH',fontsize=16)
plots.ylabel('ERROR', fontsize=16)
lists = ['Training w=5','Testing w=5','Training w=10','Testing w=10','Training w=25','Testing w=25',
         'Training w=50','Testing w=50', 'Training w=100','Testing w=100']
plots.legend(lists,fontsize=13)
plots.savefig('Zero Initialization',dpi=150, bbox_inches='tight')

ftrain_Gauss = [method[-1] for method in tr_werros]
ftest_Gauss = [method[-1] for method in te_werrors]
ftrain_Zero = [method[-1] for method in tr0_werrors]
ftest_Zero = [method[-1] for method in te0_werrors]
result = pd.DataFrame({'Width': Width,'Gaussian training error': ftrain_Gauss,'Gaussian testing error':ftest_Gauss,
'Zero init training error':ftrain_Zero, 'Zero init testing error' :ftest_Zero})

result.to_csv('Error table')

