# -*- coding: utf-8 -*-

import numpy as np

def gradientDescent(X, y, theta, step, alpha):
    m = len(y)
    theta_temp = theta
    for i in range(step):
        theta[0] -= alpha*sum(np.dot(X,theta_temp) - y) / m
        theta[1] -= alpha*sum((np.dot(X,theta_temp) - y) * X[:,1]) / m
        theta_temp = theta
    return theta