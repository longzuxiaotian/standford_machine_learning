# -*- coding: utf-8 -*-

import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np


data = scio.loadmat('./data/ex8data2.mat')
X = data['X'].copy()
plt.scatter(X[:,0],X[:,1], s = 8)
'''
m = len(X[0])

Y = np.exp(-1 * (X - u) ** 2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)
Y = np.sum(Y, axis=1) / m

x1, x2 = np.meshgrid(X[:,0].copy(), X[:,1].copy())

C = plt.contour(x1, x2, Y, 10, linewidth = 0.5)
plt.show()
'''








