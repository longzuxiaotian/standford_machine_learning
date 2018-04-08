# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import random

def kMeansInitCentroids(X, K):
    centroids = []
    for i in range(K):
        r = random.randint(1,len(X))
        centroids.append(X[r])
    return centroids

def computeDistinct(x1,x2):
    return sum((x1-x2)**2)

def findClosestCentroids(X, centroids):
    kCens = {}
    for i in range(len(centroids)):
        kCens[i] = []
    for x in X:
        relateClass = 0
        distinct = computeDistinct(x,centroids[0])
        for i in range(1,len(centroids)):
            temp = computeDistinct(x,centroids[i])
            if distinct > temp:
                relateClass = i
                distinct = temp
        kCens[relateClass].append(x)
    return kCens

def computeMeans(idx, K):
    centroids = []
    for i in range(K):
        m = len(idx[i])
        sumCen = idx[i][0].copy()
        for x in idx[i]:
            sumCen += x
        sumCen = (sumCen - idx[i][0])/m
        centroids.append(sumCen)
    return centroids

data = scio.loadmat("./data/ex7data2.mat")
X = data['X']
#X1 = data['X'][:,0]
#X2 = data['X'][:,1]

#plt.scatter(X1, X2, s=7)

K = 3

centroids = kMeansInitCentroids(X,K)
idx = findClosestCentroids(X, centroids)
#print(centroids)
#idx = findClosestCentroids(X, centroids)
#centroids = computeMeans(idx, K)


for it in range(10):
    centroids = computeMeans(idx, K)
    idx = findClosestCentroids(X, centroids)
    
cen = np.array(centroids)
for i in range(K):
    cluster = np.array(idx[i])
    plt.scatter(cluster[:,0],cluster[:,1], s = 8)
plt.scatter(cen[:,0],cen[:,1],c = 'r',marker = 'x')
plt.show()


























