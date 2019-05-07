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
train_test_ratio = 0.8
k =5
random_shuffle = 1
show_labels  = 0
num_iter = 30

dat = 5

def fst(n): 
    return n[0]  

def kernel(a,b):
    qq = LA.norm(np.subtract(a,b))
    
    return np.exp(-1*qq*qq)

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
    



if random_shuffle ==1:
     df = df.sample(frac=1)
     df.index = range(len(df.index))

for feature_name in df.columns:
        if feature_name != 'status':
         df[feature_name] = df[feature_name] - np.mean(df[feature_name].values)
         df[feature_name] = df[feature_name]/(np.max(df[feature_name].values)-np.min(df[feature_name].values))
         
X= df.values
X= X.astype('float64')


K = np.zeros((569,569))
K = K.astype('float64')
for i in range(569):
    for j in range(569):
        K[i,j] = kernel(X[i,:-1], X[j,:-1])



K1 = np.matmul((np.identity(569)-np.divide(np.ones((569,569)),569)) , K)
K2 = (np.identity(569)-np.divide(np.ones((569,569)),569))

K = np.matmul(K1, K2)

w,v = LA.eig(K)
w = [w.real]

w_sorted = sorted(w, reverse = True)
w_sorted  = np.asarray(w_sorted[0])

y = []
tot = np.sum(np.abs(w_sorted))

for i in range(569):
    y += [np.sum(np.abs(w_sorted[:i+1]))/tot]

        
x = np.arange(569)+1

plt.plot(x,y)
plt.xlabel('Number of Components', fontsize=18)
plt.ylabel('Variance Captured', fontsize=18)

# 24 components reqd
a = np.argsort(-1*(np.asarray(w)))
a = a[0]
C1 = v[a[0],:]

C2 = v[a[1],:]

C3 = v[a[2],:]    

f1 = np.matmul(X.T,C1)
f2 = np.matmul(X.T,C2)
f3 = np.matmul(X.T,C3)     
plt.plot(f1)

    



