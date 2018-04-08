# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


data1 = scio.loadmat("file/ex6data1.mat")
data2 = scio.loadmat("file/ex6data2.mat")
data3 = scio.loadmat("file/ex6data3.mat")

X = data2['X']
y = data2['y'] * 2 - 1
y = np.ravel(y)

# fit the model, don't regularize for illustration purposes

#clf = svm.SVC(kernel='linear', C=1000)
clf = svm.SVC(kernel= 'rbf',C=10000, gamma=5)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none')
plt.show()