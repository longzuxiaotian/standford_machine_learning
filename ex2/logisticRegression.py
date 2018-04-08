# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def sigmod(z):
    g = 1/(1+np.e**z)
    return g

def h(X,theta):
    return sigmod(-1 * np.dot(X,theta))

def costFuction(X,y,theta):
    m = len(y)
    J = -1/m*sum(y * np.log(h(X,theta)) + (1-y) * np.log(1-h(X,theta)))
    return J

def gradientDescent(X,y,alpha,theta,step):
    m = len(y)
    for i in range(step):
        theta -= alpha / m * np.dot(X.T,h(X,theta) - y)
    return theta

data = np.genfromtxt('ex2data1.txt',delimiter = ',')
length = len(data)
y = data[:,2]
X = np.column_stack((np.ones(length),data[:,[0,1]]))
theta = np.array([0.,0.,0.])
alpha = 0.0012
step = 100000
#print(X.shape)
new_theta = gradientDescent(X,y,alpha,theta,step)
print(costFuction(X,y,theta))
#print(new_theta)
new_X1 = data[:,1]
new_X2 = -new_theta[1]/new_theta[2] * new_X1 - new_theta[0]/new_theta[2]

x1 = data[data[:,2] == 1,:]
x2 = data[data[:,2] == 0,:]
plt.scatter(x1[:,0],x1[:,1],s = 10,c = 'g')
plt.scatter(x2[:,0],x2[:,1],s = 10,c = 'r',marker='x')
plt.plot(new_X1,new_X2)
plt.show()
'''
plt.plot(range(step),cost)
'''

