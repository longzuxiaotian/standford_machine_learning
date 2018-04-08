# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.io as scio

from sklearn.cluster import KMeans

data = scio.loadmat("./data/ex7data2.mat")
X = data['X']

y_pred = KMeans(n_clusters=3).fit_predict(X)

#print(y_pred.cluster_centers_)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s = 8)
plt.title("Kmeans")
plt.show()
