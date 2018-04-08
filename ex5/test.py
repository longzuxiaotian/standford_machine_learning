
# -*- coding: utf-8 -*-

import numpy as np
A = [[1,2,3],[4,5,6],[7,8,9]]
B = [[7,8,9],[4,5,6],[1,2,3]]
a = np.array((A,B))
b = np.array((A,B))
#c = np.array(a) + np.array(b)
a[0]=B
print(a)

x = np.mat([1, 2])
y = np.array([[1,2],[10, 20]])
yy = y[:]
yy[0][0] = 2
print(y)

m = np.array([1,2,3])
n = np.append(0,m[1:])
n[1] = 10
print(m)