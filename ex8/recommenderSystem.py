# -*- coding: utf-8 -*-

import scipy.io as scio
import numpy as np
from sklearn import linear_model

data = scio.loadmat('./data/ex8_movies.mat')
params = scio.loadmat('./data/ex8_movieParams.mat')
Y = data['Y'].copy()
R = data['R'].copy()

X = params['X'].copy()
Theta = params['Theta'].copy()
f = params['num_features'].copy()
print(f)

ans = []
'''
for i in range(100):
    ans.append(X)
    regr = linear_model.LinearRegression()
    regr.fit(X,Y)
    X = regr.get_params()
    ans.append(X)

np.savetxt('paramsX.txt',ans[-2])
np.savetxt('paramsTheta.txt',ans[-1])
'''

regr = linear_model.LinearRegression()
regr.fit(X,Y)
X = regr.coef_
print(X.shape)







