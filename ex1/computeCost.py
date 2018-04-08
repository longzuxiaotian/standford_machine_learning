# -*- coding: utf-8 -*-

import numpy as np

def computeCost(X, y, theta):
    m = len(y)
    J = sum((np.dot(X,theta) - y)**2) / (2*m)
    return J