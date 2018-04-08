# -*- coding: utf-8 -*-

import numpy as np

def costFunction(X, y, theta, lam):
    m = len(y)
    new_X = np.column_stack((np.ones(m),X))
    theta_1 = np.row_stack((np.zeros((1,1)),theta[1:]))
    J = 1/(2*m)*np.sum((np.dot(new_X,theta) - y)**2)  \
    + lam/(2*m)*np.sum(theta_1**2)
    grad = 1/m*np.dot(new_X.T,(np.dot(new_X, theta) - y)) + lam/m*theta_1
    return [J,grad]

def mathSolution(X,y):
    x = np.matrix(np.column_stack((np.ones(len(y)),X)))
    yy = np.matrix(y)
    return  np.array((x.T * x) ** -1 * x.T * yy)