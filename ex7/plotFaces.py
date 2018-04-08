# -*- coding: utf-8 -*-

import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.image as mpim

data = scio.loadmat('./data/ex7faces.mat')
X = data['X'].copy()
print(X[0])
testimg = []
for i in range(4):
    testimg.append(X[i].copy().reshape(32,32).T)

plt.subplot2grid((2,2),(0,0))
plt.imshow(testimg[0],cmap='gray')
plt.axis('off')
plt.subplot2grid((2,2),(0,1))
plt.imshow(testimg[1],cmap='gray')
plt.axis('off')
plt.subplot2grid((2,2),(1,0))
plt.imshow(testimg[2],cmap='gray')
plt.axis('off')
plt.subplot2grid((2,2),(1,1))
plt.imshow(testimg[3],cmap='gray')
plt.axis('off')