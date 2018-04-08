# -*- coding: utf-8 -*-

import scipy.io as scio
import matplotlib.pyplot as plt
import linearRegCostFunction as lrcf
import numpy as np

def getTheta(X,y,theta,step,alpha,lam):
    theta_new = theta.copy()
    for i in range(step):
        [J, grad] = lrcf.costFunction(X, y, theta_new, lam)
        theta_new = theta_new - alpha*grad
    return theta_new


data = scio.loadmat("ex5data1.mat")
#print(type(data))
print(data.keys())

X = data['X']
y = data['y']
'''
plt.scatter(X,y,s=15)
plt.title('data')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
'''

'''
=============== part 1 ===============
计算代价函数和梯度值,画出线性回归函数
'''

theta = np.array([[1],[1]])
lam = 0
#[J,grad] = lrcf.costFunction(X, y, theta, lam)
#t_math = np.array(lrcf.mathSolution(X, y))
#print(lrcf.costFunction(X, y, t_math, lam)[0])
#print(J)
#print(grad)

alpha = 0.001
step = 1
for i in range(step):
    [J, grad] = lrcf.costFunction(X, y, theta, lam)
    theta = theta - alpha*grad

a = range(-50,50,1)
b = a * theta[1] + theta[0]

plt.title('linear regression - part 1 -')
plt.xlabel('X')
plt.ylabel('y')
plt.scatter(X,y,s=15)
plt.plot(a,b)
plt.show()


'''
=============== part 2 ================
Bias-variance
学习曲线
high bias 的情况
'''

Xval = data['Xval']
yval = data['yval']

Jtrain = []
Jval = []

for n in range(1,len(X)+1):
    jtrain = 0
    tempX = X[range(n),:]
    tempy = y[range(n),:]
    '''
    for i in range(step):
        grad = lrcf.costFunction(tempX, tempy, theta2, lam)[1]
        theta2 = theta2 - alpha*grad
    '''
    theta2 = lrcf.mathSolution(tempX,tempy);
    jtrain = lrcf.costFunction(tempX, tempy, theta2, lam)[0]
    Jtrain.append(jtrain)
    
    vX = Xval[range(n),:]
    vy = yval[range(n),:]
    jval = lrcf.costFunction(vX, vy, theta2, lam)[0]
    Jval.append(jval)
    
plt.plot(range(len(Jtrain)),Jtrain)
plt.plot(range(len(Jval)),Jval)
plt.show()    

'''
============== part 3 ===============
Ploynomial regression
'''
def pretreatment(data):
    middle = np.mean(data, axis = 0)
    sigma = np.std(data, axis = 0)
    X = (data - middle) / sigma
    return [X,middle,sigma]


ploy = 8
ploy_X = X.copy()
for i in range(2,ploy+1):
    ploy_X = np.column_stack([ploy_X,X**i])

[new_ploy_X,middle,sigma] = pretreatment(ploy_X)

#print(new_ploy_X)

theta3 = np.zeros((ploy+1,1))
step3 = 10000
alpha3 = 0.008
lam3 = 0
new_theta3 = getTheta(new_ploy_X, y , theta3, step3, alpha3, lam3)

#J = lrcf.costFunction(new_ploy_X, y, new_theta3, lam)[0]
#print(J)

a3 = np.array(range(-60,60,1))

b3 = np.zeros(len(a3))
for i in range(1,ploy+1):
    b3 = b3 + (a3**i-middle[i-1])/sigma[i-1] * new_theta3[i] 
b3 = b3 + new_theta3[0]

#print(b3)
plt.plot(a3,b3)
plt.scatter(X,y,marker = 'x',c = 'r')
plt.show()

#Xval = data['Xval']
#yval = data['yval']

Jtrain3 = []
Jval3 = []

for n in range(2,len(X)+1):
    jtrain = 0
    tempx = X[range(n),:]
    ploy_X = tempx.copy()
    for i in range(2,ploy+1):
        ploy_X = np.column_stack([ploy_X,tempx**i])
    
    [new_tempX,middle,sigma] = pretreatment(ploy_X)
    tempy = y[range(n),:]

    new_theta3 = getTheta(new_tempX, tempy , theta3, step3, alpha3, lam3)
    jtrain = lrcf.costFunction(new_tempX, tempy, new_theta3, lam3)[0]
    Jtrain3.append(jtrain)
    
    vX = Xval[range(n),:]
    vy = yval[range(n),:]
    
    tempvX = Xval[range(n),:]
    ploy_vX = tempvX.copy()
    for i in range(2,ploy+1):
        ploy_vX = np.column_stack([ploy_vX,tempvX**i])
    
    newvX = (ploy_vX - middle)/sigma
    tempvy = yval[range(n),:]
    
    jval = lrcf.costFunction(newvX, tempvy, new_theta3, lam3)[0]
    Jval3.append(jval)

plt.plot(range(2,len(Jtrain3)+2),Jtrain3)
plt.plot(range(2,len(Jval3)+2),Jval3)
plt.show()













