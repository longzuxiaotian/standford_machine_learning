# -*- coding: utf-8 -*-

import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

data = scio.loadmat('./data/ex7faces.mat')
preX = data['X'].copy()
principal = 100

pca = PCA(n_components=principal)
X_r = pca.fit_transform(preX)

#print(eigenfaces)
#print(X_r.shape)

returnimg = pca.inverse_transform(X_r)
testimg = returnimg[0].copy().reshape(32,32).T
plt.imshow(testimg,cmap='gray')
plt.axis('off')
