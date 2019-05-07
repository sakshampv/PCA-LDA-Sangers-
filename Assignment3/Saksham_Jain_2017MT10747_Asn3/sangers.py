# =============================================================================
# import numpy as np
# 
# from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# import numpy as np
# import random
# import warnings
# import math
# from numpy import linalg as LA
# import matplotlib.pyplot as plt
# from sklearn.naive_bayes import MultinomialNB
# from sklearn import preprocessing
# from sklearn.naive_bayes import GaussianNB
# 
# 
# #########################################################################
# train_test_ratio = 0.8
# k =5
# random_shuffle = 1
# show_labels  = 0
# num_iter = 30
# 
# dat = 5
# 
# def fst(n): 
#     return n[0]  
# 
# warnings.filterwarnings("ignore")
# 
# if dat == 5:
#     df = pd.read_table('./Breast_Cancer/wdbc.data', delimiter = ',',header= None, names=['id', 'status', 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
#     df = df.drop(['id'], axis = 1)
#     for index, row in df.iterrows():
#         if df.loc[index]['status'] == 'M':
#              df.set_value(index,'status', 0)
#         if df.loc[index]['status'] == 'B':
#              df.set_value(index,'status', 1)     
#     df = df[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,'status']]              
#     
# 
# 
# 
# if random_shuffle ==1:
#      df = df.sample(frac=1)
#      df.index = range(len(df.index))
# 
# for feature_name in df.columns:
#         if feature_name != 'status':
#          df[feature_name] = df[feature_name] - np.mean(df[feature_name].values)
#          df[feature_name] = df[feature_name]/(np.var(df[feature_name].values))
#          
# X= df.values
# X= X.astype('float64')
# 
# # Set random seed for reproducibility
# np.random.seed(1000)
# 
# scaler = StandardScaler(with_std=False)
# Xs = scaler.fit_transform(X)
# 
# # Compute eigenvalues and eigenvectors
# Q = np.cov(Xs.T)
# eigu, eigv = np.linalg.eig(Q)
# 
# W_sanger = np.random.normal(scale=0.1, size=(31, 31))
# prev_W_sanger = np.ones((31, 31))
# 
# learning_rate = 0.1
# nb_iterations = 2000
# t = 0.0
# 
# for i in range(nb_iterations):
#     prev_W_sanger = W_sanger.copy()
#     dw = np.zeros((31, 31))
#     t += 1.0
#     
#     for j in range(Xs.shape[0]):
#         Ysj = np.dot(W_sanger, Xs[j]).reshape((31, 1))
#         QYd = np.tril(np.dot(Ysj, Ysj.T))
#         dw += np.dot(Ysj, Xs[j].reshape((1, 31))) - np.dot(QYd, W_sanger)
#         
#     W_sanger += (learning_rate / t) * dw
#     W_sanger /= np.linalg.norm(W_sanger, axis=1).reshape((31, 1))
#     
# 
# #C = (1/569)*np.matmul(X.T, X)
#     
# y = []
# tot = np.sum(np.abs(eigu))
# 
# for i in range(31):
#     y += [np.sum(np.abs(eigu[:i+1]))/tot]
# 
#         
# x = np.arange(31)+1
# =============================================================================

plt.plot(x[:10],y[:10])
plt.xlabel('Number of Components', fontsize=18)
plt.ylabel('Variance Captured', fontsize=18)
fig.savefig('ssq.jpg')
# =============================================================================
# 
# plt.plot(x,y)
# plt.xlabel('Number of Components', fontsize=18)
# plt.ylabel('Variance Captured', fontsize=18)
# fig.savefig('ssq11.jpg')
# 
#     
# =============================================================================
    



    
    