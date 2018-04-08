# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

data1 = scio.loadmat("file/ex6data1.mat")
data2 = scio.loadmat("file/ex6data2.mat")
data3 = scio.loadmat("file/ex6data3.mat")

X1 = data2['X']
y1 = data2['y'] * 2 - 1
y1 = np.ravel(y1)

plt.scatter(X1[:, 0], X1[:, 1], c=y1, s=15)
plt.show()