# -*- coding: utf-8 -*-

import scipy.io as scio
import numpy as np

def sigmoid(z) :
    return 1.0 / (1.0 + np.e ** (-1*z))

#theta * x
data = scio.loadmat('ex3data1.mat')
parameters = scio.loadmat('ex3weights.mat')
y = data['y']
X = data['X']
m = len(y)
theta1 = parameters['Theta1']
theta2 = parameters['Theta2']
count = 0

for i in range(m):
    temp1 = np.append(1,X[i])
    temp2 = np.append(1,sigmoid(np.dot(theta1,temp1)))
    res = sigmoid(np.dot(theta2,temp2))
    for j in res:
        if j>0.5:
            count +=1
print(count)
print(m)
print(theta2)
#print(theta1.shape)
#print(np.append(1,X[0])[0])