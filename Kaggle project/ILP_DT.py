import sys
import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

trainData = pd.read_csv('/Users/u1368460/Documents/Machine Learning/PROJECT/ilp2021f /train_final.csv')
testData = pd.read_csv('/Users/u1368460/Documents/Machine Learning/PROJECT/ilp2021f /test_final.csv')


label = preprocessing.LabelEncoder()
label.fit(trainData['age'].unique())
trainData['age']=label.transform(trainData['age'])
label.fit(trainData['fnlwgt'].unique())
trainData['fnlwgt']=label.transform(trainData['fnlwgt'])
label.fit(trainData['education'].unique())
trainData['education']=label.transform(trainData['education'])
label.fit(trainData['education.num'].unique())
trainData['education.num']=label.transform(trainData['education.num'])
label.fit(trainData['marital.status'].unique())
trainData['marital.status']=label.transform(trainData['marital.status'])
label.fit(trainData['occupation'].unique())
trainData['occupation']=label.transform(trainData['occupation'])
label.fit(trainData['relationship'].unique())
trainData['relationship']=label.transform(trainData['relationship'])
label.fit(trainData['race'].unique())
trainData['race']=label.transform(trainData['race'])
label.fit(trainData['sex'].unique())
trainData['sex']=label.transform(trainData['sex'])
label.fit(trainData['capital.gain'].unique())
trainData['capital.gain']=label.transform(trainData['capital.gain'])
label.fit(trainData['capital.loss'].unique())
trainData['capital.loss']=label.transform(trainData['capital.loss'])
label.fit(trainData['hours.per.week'].unique())
trainData['hours.per.week']=label.transform(trainData['hours.per.week'])
label.fit(trainData['native.country'].unique())
trainData['native.country']=label.transform(trainData['native.country'])
label.fit(trainData['workclass'].unique())
trainData['workclass']=label.transform(trainData['workclass'])

label.fit(testData['age'].unique())
testData['age']=label.transform(testData['age'])
label.fit(testData['workclass'].unique())
testData['workclass']=label.transform(testData['workclass'])
label.fit(testData['fnlwgt'].unique())
testData['fnlwgt']=label.transform(testData['fnlwgt'])
label.fit(testData['education'].unique())
testData['education']=label.transform(testData['education'])
label.fit(testData['education.num'].unique())
testData['education.num']=label.transform(testData['education.num'])
label.fit(testData['marital.status'].unique())
testData['marital.status']=label.transform(testData['marital.status'])
label.fit(testData['occupation'].unique())
testData['occupation']=label.transform(testData['occupation'])
label.fit(testData['relationship'].unique())
testData['relationship']=label.transform(testData['relationship'])
label.fit(testData['race'].unique())
testData['race']=label.transform(testData['race'])
label.fit(testData['sex'].unique())
testData['sex']=label.transform(testData['sex'])
label.fit(testData['capital.gain'].unique())
testData['capital.gain']=label.transform(testData['capital.gain'])
label.fit(testData['capital.loss'].unique())
testData['capital.loss']=label.transform(testData['capital.loss'])
label.fit(testData['hours.per.week'].unique())
testData['hours.per.week']=label.transform(testData['hours.per.week'])
label.fit(testData['native.country'].unique())
testData['native.country']=label.transform(testData['native.country'])

classifierFunc = tree.DecisionTreeClassifier()
classifierFunc.fit(trainData.drop(['income>50K'],axis=1),trainData['income>50K'])
output=classifierFunc.predict(testData.drop(['ID'],axis=1))
output_col=classifierFunc.predict_proba(testData.drop(['ID'],axis=1))

ID=[]
for j in range(23842):
    ID.append(j+1)

output_data_col = {'ID':ID,"Prediction":output_col[:,1]}
output_data=pd.DataFrame(output_data_col)
output_data.to_csv("MLsub_DT.csv",index=False)

