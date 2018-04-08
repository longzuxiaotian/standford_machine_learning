# -*- coding: utf-8 -*-

import numpy as np

def gradientDescent(X, y, theta, step, alpha):
    m = len(y)
    for i in range(step):
        theta -= alpha / m * np.dot(X.T,np.dot(X,theta) - y)
    return theta