# -*- coding: utf-8 -*-

import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

data = scio.loadmat("./data/ex7data1.mat")
preX = data['X'].copy()

meanX = np.mean(preX,axis=0)
X = preX - meanX
#plt.scatter(X[:,0],X[:,1],s = 8)
#print(X)
covarianceMatrix = np.dot(X.T,X)
u,sigma,vt = np.linalg.svd(covarianceMatrix)
sigma2, p = np.linalg.eig(covarianceMatrix)

x = u[0].copy()
x.shape = (len(x),1)
oneD = np.dot(X,x)
#print(oneD)
pca = PCA(n_components=2)
X_r = pca.fit_transform(preX)
#print(X_r)
aaa = X_r + preX
print(aaa)
#plt.plot(oneD[:,0],oneD[:,1])
#plt.scatter(returnX[:,0],returnX[:,1],c = 'g')

















