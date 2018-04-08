# -*- coding: utf-8 -*-

import numpy as np
from lrCostFuction import lrCostFuction

def oneVsAll(X, y, num_lables, lam):
    m = len(X)
    n = len(X[0])
    
    all_theta = np.zeros((num_lables, n+1))
    X = np.column_stack((np.ones(m), X))
    initial_theta = np.zeros(n+1)
    alpha = 0.0001
    step = 2000
    
    new_y = np.zeros(m)
    all_theta = np.zeros((1,n+1))
    for j in range(m) :
        new_y[j] = int(y[j] == num_lables)
    all_theta = getTheta(X, new_y, m, lam, initial_theta, step, alpha)
    return all_theta

def getTheta(X, y, m, lam, theta, step, alpha):
    #J = np.zeros(setp)
    res = theta[:]
    for i in range(step):
        [J,grad] = lrCostFuction(res, X, y, lam)
        res -= alpha * grad
        #print(costFuction(X,y,np.append(theta_0,theta_o),lamda))
    return res