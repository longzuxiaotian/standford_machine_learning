# -*- coding: utf-8 -*-

import numpy as np
from sigmoid import sigmoid

def lrCostFuction(theta, X, y, lam):
    m = len(y)
    J = 0
    grad = np.zeros(len(theta))
    temp = np.array(np.append(0,theta[1:len(theta)]))
    J = -1 * sum(y*np.log(sigmoid(np.dot(X,theta))) + (1-y)* \
                 np.log(1-sigmoid(np.dot(X,theta)))) / m + \
                 lam/(2*m) * temp * temp
    grad = np.dot(X.T ,(sigmoid(np.dot(X,theta)) - y)) / m + lam/m * temp
    return [J,grad]