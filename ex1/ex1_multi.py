# -*- coding: utf-8 -*-

import numpy as np
from pretreatment import pretreatment
from gradientDescent_multi import gradientDescent
from math_solution import math_solution

data = np.genfromtxt('ex1data2.txt',delimiter = ',')
#print(data)
m = len(data)
pre_X, middle, sigma = pretreatment(data[:,[0,1]])
X = np.column_stack((np.ones(m).T,pre_X))
#print(X)
y = data[:,2]

theta = np.array([0.,0.,0.])
alpha = 0.01
step = 2000
'''
theta_new = gradientDescent(X, y, theta, step, alpha)
print(theta_new)
'''
xx = np.matrix(X)
yy = np.matrix(y)
#print(xx.shape)
#print(yy.T.shape)
theta_new = math_solution(xx, yy.T)
print(theta_new)

