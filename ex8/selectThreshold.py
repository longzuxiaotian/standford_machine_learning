# -*- coding: utf-8 -*-

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

def computeP(X):
    m = len(X)
    u = np.mean(X, axis = 0)
    sigma2 = np.sum((X-u)**2, axis = 0) / (m-1)
    pval = np.exp(-1 * (X-u) ** 2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)
    pval = np.prod(pval, axis = 1)
    return pval

def selectTreshold(filename = './data/ex8data2.mat'):
    data = scio.loadmat(filename)
    Xval = data['Xval'].copy()
    pval = computeP(Xval)
    yval = data['yval'].copy()
    combine = np.column_stack((pval,yval))
    
    F1score = 0
    tre = 0
    #预测的类别为1，真实为1
    tp = 0
    #预测的类别为1，真实为0
    fp = 0
    #预测的类别为0，真实为1
    fn = 0
    stepsize = (max(pval) - min(pval)) / 1000
    for e in np.arange(min(pval),max(pval),stepsize):
        a = combine[combine[:,0]<e]
        b = combine[combine[:,1] == 1]
        tp = len(a[a[:,1] == 1])
        if tp == 0:
            continue
        prec = tp/len(a)
        rec = tp/len(b)
        temp = 2*prec*rec / (prec + rec)
        if temp > F1score :
            F1score = temp
            tre = e
    #plt.scatter(Xval[:,0],Xval[:,1])
    return tre, F1score

if __name__ == "__main__":
    #selectTreshold()
    tre, F1score = selectTreshold()
    print(F1score)
    print(tre)






