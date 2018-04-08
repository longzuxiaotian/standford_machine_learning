# -*- coding: utf-8 -*-

import oneVsAll as ova
import scipy.io as scio
import numpy as np

data = scio.loadmat("ex3data1.mat")
X = data['X'] * 255
y = data['y']
#print(X.shape)
allt = ova.oneVsAll(X, y, 2, 0)
count = 0
for i in range(5000):
    isone = np.dot(allt,np.array(np.append(1,X[i])))
    if isone > 0 :
        count += 1
print(count)