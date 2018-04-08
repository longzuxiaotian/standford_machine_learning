# -*- coding: utf-8 -*-

import numpy as np

a = np.array([[1,2],[1,3],[2,3]])
aa = np.array([[0,2],[1,0],[0,3]])

b = np.array([1,2])
c = a.copy()
d = [[1,2,3],1]
e = d.copy()
e[0][0] = 2
print(a[aa[:] == 0])