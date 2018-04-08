# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as scio
from nnCostFunction import gradientTheta
from nnCostFunction import costFunction
from oneCostFunction import oneCostFunction

data = scio.loadmat('ex3data1.mat')
parameters = scio.loadmat('ex3weights.mat')
X = data['X']
y = data['y']
theta = np.array([parameters['Theta1'],parameters['Theta2']])
K = 10


#J = costFunction(X,y,K,theta)
#J = testCostFunction(X[1000],y[1000],theta)

#print(theta[1])

def random_theta():
    t = [None]*2
    [m1,n1] = theta[0].shape
    [m2,n2] = theta[1].shape
    
    t[0] = 1 - 2*np.random.random((m1,n1))
    t[1] = 1 - 2*np.random.random((m2,n2))
    return t

step = 50
alpha = 0.1

init_t = random_theta()
random_init_theta = np.array(init_t)
#print(random_init_theta[0].shape)
new_theta = gradientTheta(X, y, K, theta-0.05, step, alpha)
#print(costFunction(X,y,K,new_theta))








