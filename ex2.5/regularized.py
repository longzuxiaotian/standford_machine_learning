# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plotData(data):
    x1 = data[data[:,2] == 1,:]
    x2 = data[data[:,2] == 0,:]
    plt.scatter(x1[:,0],x1[:,1],s = 10,c = 'r',marker = 'x')
    plt.scatter(x2[:,0],x2[:,1],s = 10,c = 'g')
    plt.show()

def pretreatment(data):
    middle = np.mean(data, axis = 0)
    sigma = np.std(data, axis = 0)
    X = (data - middle) / sigma
    return [X,middle,sigma]

def sigmod(z):
    g = 1/(1+np.e**z)
    return g

def h(X,theta):
    return sigmod(-1 * np.dot(X,theta))

def costFuction(X,y,theta,lamda):
    m = len(y)
    J = -1/m*sum(y * np.log(h(X,theta)) + (1-y) * np.log(1-h(X,theta))) + lamda/(2*m)*sum(theta**2)
    return J

def gradientDescent(X,y,alpha,theta,step,lamda):
    m = len(y)
    theta_0 = theta[0]
    theta_o = theta[1:28]
    for i in range(step):
        theta_0 -= alpha / m * np.dot(X[:,0],sigmod(-1*X[:,0]*theta_0) - y)
        theta_o -= alpha / m * np.dot(X[:,range(1,28)].T,h(X[:,range(1,28)],theta_o) - y) + lamda/m*theta_o
        #print(costFuction(X,y,np.append(theta_0,theta_o),lamda))
    return np.append(theta_0,theta_o)

data = np.genfromtxt('ex2data2.txt',delimiter = ',')
length = len(data)
y = data[:,2]
#X = data[:,[0,1]]
#[new_X, middle, sigma] = pretreatment(X)
#print(new_X)
x1 = data[:,0]
x2 = data[:,1]
X = np.ones((length,28))
size = 1
for i in range(1,7):
    for j in range(i+1):
        X[:,size] = ( x1**(i-j) ) * (x2**j)
        size += 1
#print(X[:,range(1,28)])
        
theta = np.zeros(size)
alpha = 0.01
step = 100000

new_theta = gradientDescent(X,y,alpha,theta,step,1)
print(new_theta)
predict_y = new_theta[0]
s = 1
for i in range(1,7):
    for j in range(i+1):
        predict_y += ( -0.5**(i-j) ) * (0.7**j) * new_theta[s]
        s += 1
print(sigmod(-1*predict_y))
