import pandas as pd
import numpy as np
import random
import warnings
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB


#########################################################################

dat = 5

def fst(n): 
    return n[0]  

warnings.filterwarnings("ignore")

if dat == 5:
    df = pd.read_table('./Breast_Cancer/wdbc.data', delimiter = ',',header= None, names=['id', 'status', 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
    df = df.drop(['id'], axis = 1)
    for index, row in df.iterrows():
        if df.loc[index]['status'] == 'M':
             df.set_value(index,'status', 0)
        if df.loc[index]['status'] == 'B':
             df.set_value(index,'status', 1)     
    df = df[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,'status']]              

for feature_name in df.columns:
        if feature_name != 'status':
         df[feature_name] = df[feature_name] - np.mean(df[feature_name].values)
         df[feature_name] = df[feature_name]/(np.var(df[feature_name].values))
         
X= df.values
X= X.astype('float64')

X0 = X[np.where(X[:,-1]==0)]
X1 = X[np.where(X[:,-1]==1)]

myu0 =np.mean(X0, axis = 0)
myu1 =np.mean(X1, axis = 0)
A = myu1 - myu0
A = np.asmatrix(A)
Sb = np.matmul(A.T, A)
Sw = np.matmul(X1.T, X1) + np.matmul(X0.T, X0)
SwI = np.linalg.inv(Sw)
A = np.matmul(SwI, Sb)

w, v = LA.eig((A+A.T)/2)

df22 = pd.DataFrame(v)
df22.to_csv('df222.csv')
v = (np.conj(v) + v)/2
v = v.real

    

f1 = np.matmul(X,np.asarray(v[0,:]).T)
f2 = np.matmul(X,np.asarray(v[1,:]).T)
f3 = np.matmul(X,np.asarray(v[2,:]).T) 
f4 = np.matmul(X,np.asarray(v[3,:]).T)
f5 = np.matmul(X,np.asarray(v[4,:]).T)  

X2  = np.column_stack((f1,f2))
X3  =np.column_stack((f1,f2,f3))
X4 = np.column_stack((f1,f2,f3,f4))
X5 = np.column_stack((f1,f2,f3,f4,f5))

y_train = X[:,-1]
X1 = preprocessing.scale(X1)
X2 = preprocessing.scale(X2)
X3 = preprocessing.scale(X3)
X4 = preprocessing.scale(X4)
X5 = preprocessing.scale(X5)
# =============================================================================
# model = GaussianNB().fit(X1, y_train)  
# predicted = model.predict(X1)
# print('\n')
# print(np.mean(predicted == y_train))  
# 
# 
# =============================================================================
model = GaussianNB().fit(X2, y_train)  
predicted = model.predict(X2)
print('\n')
print(np.mean(predicted == y_train))

model = GaussianNB().fit(X3, y_train)  
predicted = model.predict(X3)
print('\n')
print(np.mean(predicted == y_train))


model = GaussianNB().fit(X4, y_train)  
predicted = model.predict(X4)
print('\n')
print(np.mean(predicted == y_train)) 

model = GaussianNB().fit(X5, y_train)  
predicted = model.predict(X5)
print('\n')
print(np.mean(predicted == y_train)) 













