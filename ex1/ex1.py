# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from computeCost import computeCost
from gradientDescent import gradientDescent

data = np.genfromtxt('ex1data1.txt',delimiter = ',');
m = len(data)
#data1 = np.matrix(data)
data1 = np.column_stack((np.ones(m).T,data))
X = data1[:,[0,1]]
y = data1[:,2]

plt.scatter(X[:,1],y)

theta = np.array([0.,1.])
step = 1500
alpha = 0.01
theta_new = gradientDescent(X, y, theta, step, alpha)
plt.plot(X[:,1],X[:,1]*theta_new[1]+theta_new[0],color = 'red')
plt.show()
print(computeCost(X,y,theta))