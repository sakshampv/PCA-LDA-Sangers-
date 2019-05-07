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



### PREPROCESSING  
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
C = np.matmul(X.T,X)/(len(df.index))

w, v = LA.eig(C)   
w = [w.real]
aaa = [w,v]
q = np.zeros((31,32), dtype = 'float64')
q[:,:-1] = v
q[:,31] = np.asarray(w)
#q = sorted(q, key=lambda a_entry: a_entry[569], reverse = True) 
s = zip(w,v)
s=sorted(s, reverse = True)
w = sorted(w, reverse = True)
w = np.asarray(w)
w = w.T
d1 = pd.DataFrame(q)
d1.to_csv('dff.csv')
x = np.arange(31) + 1
y = []
tot = np.sum(np.abs(w))

for i in range(31):
    y += [np.sum(np.abs(w[:i+1]))/tot]

fig = plt.figure()


plt.plot(x[:5],y[:5])
plt.xlabel('Number of Components', fontsize=18)
plt.ylabel('Variance Captured', fontsize=18)
fig.savefig('ssq.jpg')

#plt.plot(x,y)
plt.xlabel('Number of Components', fontsize=18)
plt.ylabel('Variance Captured', fontsize=18)
fig.savefig('ssq11.jpg')


# 75
# first

we = np.asarray(q[0,:-1])


f1 = np.matmul(X,we.T)
f2 = np.matmul(X,np.asarray(q[1,:-1]))
f3 = np.matmul(X,np.asarray(q[2,:-1])) 
f4 = np.matmul(X,np.asarray(q[3,:-1]))
f5 = np.matmul(X,np.asarray(q[4,:-1]))  

#plt.plot(f1,f2,f3)
X2  = np.column_stack((f1,f2))
X3  =np.column_stack((f1,f2,f3))
X4 = np.column_stack((f1,f2,f3,f4))
X5 = np.column_stack((f1,f2,f3,f4,f5))

y_train = X[:,-1]

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


