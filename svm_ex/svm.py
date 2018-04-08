# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#加载数据
data1 = np.genfromtxt("ex2data1.txt", delimiter = ',')
X = data1[:,[0,1]]
plot1 = data1[data1[:,2] == 0,:]
plot2 = data1[data1[:,2] == 1,:]
y = data1[:,2] * 2 - 1
size = len(y)

#定义向量机参数 w*x + b
alphas = np.zeros((size,1))
b = 0 
error = np.zeros((size,2))
C = 0.6 # C 松弛变量影响因子
it = 0 # 遍历个数
tol = 0.001

flag = 1 # 作为判断是否全遍历的标记
alphas_change = 0
'''
启发式选择两个alpha
'''

def calEk(X, alphas, y, b, i):
    pre_Li = sum((alphas*y) * np.dot(X,X[i,:].T)) +b
    return pre_Li - y[i]

def select(i,data,num_data,alphas,label,b,C,Ei,choose):
    maxDeltaE = 0
    maxJ = -1
    if choose == 1:
        j = np.random.randint(num_data)
        if j == i:
            temp = 1
            while temp:
                j = np.random.randint(num_data)
                if j != i:
                    temp = 0
        J = j
        Ej = calEk(data,alphas,label,b,J)
    else:
        temp = []
        for alp in alphas:
            if alp>0 and alp<C:
                temp.append()
        index = np.array(temp)
        for k in range(len(index)):
            if i == index[k]:
                continue
            temp_e = calEk(data,alphas,label,b,k)
            deltaE = abs(Ei - temp_e)
            if deltaE > maxDeltaE:
                maxJ = k
                maxDeltaE = deltaE
                Ej = temp_e
        J = maxJ
    return [J,Ej]
while it<size and ((alphas_change>0) or flag):
    alphas_change = 0
    # 当flag = 1 时，遍历所有样本
    if flag:
        for i in range(size):
            Ei = calEk(X, alphas, y, b, i)
            # 判断是否满足KKT条件
            if (y[i]*Ei<-tol and alphas[i]<C) or \
            (y[i]*Ei>tol and alphas[i]>0):
                # 选择第一个alpha
                [j,Ej] = select(i,X,size,alphas,y,b,C,Ei,flag)
                alphas_i_old = alphas[i]
                alphas_j_old = alphas[j]
                if y[i] != y[j]:
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C + alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i] -C)
                    H = min(C,alphas[j] + alphas[i])
                if L == H:
                    continue
                eta = 2*sum(X[i,:]*X[j,:]) - sum(X[i,:]**2) - sum(X[j,:]**2)
                if eta>0:
                    continue
                alphas[j] = alphas[j] - y[j]*(Ei-Ej)/eta
                # 限制范围
                if alphas[j] > H:
                    alphas[j] = H
                elif alphas[j] < L:
                    alphas[j] = L
                if abs(alphas[j] - alphas_j_old) < 1e-4:
                    continue
                alphas[i] = alphas[i] + y[i]*y[j]*(alphas_j_old - alphas[j])
                b1 = b - Ei - y[i]*(alphas[i]-alphas_i_old)*sum(X[i,:]**2) \
                -y[j]*(alphas[j]-alphas_j_old)*sum(X[i,:]*X[j,:])
                
                b2 = b - Ej - y[i]*(alphas[i]-alphas_i_old)*sum(X[i,:]*X[j,:]) \
                -y[j]*(alphas[j]-alphas_j_old)*sum(X[j,:]**2)
                
                if alphas[i]>0 and alphas[i]<C:
                    b = b1
                elif alphas[j]>0 and alphas[j]<C:
                    b = b2
                else:
                    b = (b1+b2)/2
                    
                alphas_change += 1
        it += 1
    else: # 遍历alpha = 0~C的样本
        temp = []
        for alp in alphas:
            if alp>0 and alp<C:
                temp.append()
        index = np.array(temp)
        for ii in range(index):
            i = index[ii]
            Ei = calEk(X, alphas, y, b, i)
            if (y[i]*Ei<-tol and alphas[i]<C) or \
            (y[i]*Ei>tol and alphas[i]>0):
                # 选择第一个alpha
                [j,Ej] = select(i,X,size,alphas,y,b,C,Ei,flag)
                alphas_i_old = alphas[i]
                alphas_j_old = alphas[j]
                if y[i] != y[j]:
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C + alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i] -C)
                    H = min(C,alphas[j] + alphas[i])
                if L == H:
                    continue
                eta = 2*sum(X[i,:]*X[j,:]) - sum(X[i,:]**2) - sum(X[j,:]**2)
                if eta>0:
                     continue
                alphas[j] = alphas[j] - y[j]*(Ei-Ej)/eta
                # 限制范围
                if alphas[j] > H:
                    alphas[j] = H
                elif alphas[j] < L:
                    alphas[j] = L
                if abs(alphas[j] - alphas_j_old) < 1e-4:
                    continue
                alphas[i] = alphas[i] + y[i]*y[j]*(alphas_j_old - alphas[j])
                b1 = b - Ei - y[i]*(alphas[i]-alphas_i_old)*sum(X[i,:]**2) \
                -y[j]*(alphas[j]-alphas_j_old)*sum(X[i,:]*X[j,:])
                
                b2 = b - Ej - y[i]*(alphas[i]-alphas_i_old)*sum(X[i,:]*X[j,:]) \
                -y[j]*(alphas[j]-alphas_j_old)*sum(X[j,:]**2)
                
                if alphas[i]>0 and alphas[i]<C:
                    b = b1
                elif alphas[j]>0 and alphas[j]<C:
                    b = b2
                else:
                    b = (b1+b2)/2
                    
                alphas_change += 1
        it += 1
    
    if flag:
        flag = 0
    elif alphas_change == 0:
        flag = 1
    
# 计算权值W
W = np.dot((alphas*y).T,X)

k = -W[1]/W[2]
x = range[15:0.1:100]
y = k*x + b
plt.plot(x,y)




























