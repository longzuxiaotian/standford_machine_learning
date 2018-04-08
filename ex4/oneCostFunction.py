# -*- coding: utf-8 -*-

import numpy as np
from sigmoid import sigmoid

def oneCostFunction(X,y,K,theta):
    ty = np.zeros(K)
    ty[y-1] = 1
    temp_a = X[:]
    a = []
    a.append(temp_a)
    z = []
    for i in range(0,len(theta)):
        temp_a = np.append(1,temp_a)
        temp_z = np.dot(theta[i],temp_a)
        z.append(temp_z)
        temp_a = sigmoid(temp_z)
        a.append(temp_a)
    grad = getGrad(a,ty,theta)
    J = sum(-ty*np.log(a[-1]) - (1-ty)*np.log(1-a[-1]))
    return [J, grad]

def getGrad(a,y,theta):
    gradTheta = [None]*len(theta)
    delta = a[-1] - y
    delta.shape = (len(delta),1)
    for i in range(len(a)-1,0,-1):
        temp_a = np.append(1,a[i-1])
        temp_a.shape = (1,len(temp_a))
        
        gradTheta[i-1] = np.dot(delta,temp_a)
        delta = np.dot(theta[i-1].T,delta) * temp_a.T * (1-temp_a.T)
        #点睛之笔！！！！！
        delta = delta[1:]
    return np.array(gradTheta)

        
        















