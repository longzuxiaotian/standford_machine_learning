# -*- coding: utf-8 -*-

import numpy as np
from sigmoid import sigmoid
from oneCostFunction import oneCostFunction


def costFunction(all_X,all_y,K,theta):
    m = len(all_y)
    J = 0
    for i in range(m):
        [j,g] = oneCostFunction(all_X[i],all_y[i],K,theta)
        J += j
    return 1/m*J

def gradientTheta(all_X, all_y, K, theta, step, alpha):
    m = len(all_y)
    for step_num in range(step):
        [J,all_grad] = oneCostFunction(all_X[0],all_y[0],K,theta)
        for exmple in range(1,m):
            [j,grad] = oneCostFunction(all_X[exmple],all_y[exmple],K,theta)
            J += j
            all_grad = all_grad + grad
        theta = theta - alpha/m*all_grad
        #print(all_grad/m)
        print(J/m)
    return theta
            