# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

data1 = np.genfromtxt("ex2data1.txt", delimiter = ',')
data2 = np.genfromtxt("ex2data2.txt", delimiter = ',')
X1a = data1[data1[:,2] == 0,:]
X1b = data1[data1[:,2] == 1,:]
y1 = data1[:,2]
X2a = data2[data2[:,2] == 0,:]
X2b = data2[data2[:,2] == 1,:]
y2 = data2[:,2]

plt.scatter(X1a[:,0],X1a[:,1], s = 15, c = 'r', marker = 'x')
plt.scatter(X1b[:,0],X1b[:,1], s = 15, c = 'g')
plt.title("data1")
plt.show()

plt.scatter(X2a[:,0],X2a[:,1], s = 15, c = 'r', marker = 'x')
plt.scatter(X2b[:,0],X2b[:,1], s = 15, c = 'g')
plt.title("data2")
plt.show()

print(np.random.randint(10))