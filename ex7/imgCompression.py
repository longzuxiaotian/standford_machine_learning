# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import copy

from sklearn.cluster import KMeans

#im = mpimg.imread("./data/bird_small.png")
im = mpimg.imread("./data/owntest.jpg")
x = len(im)
y = len(im[0])
#plt.imshow(im)

data = copy.deepcopy(im)
#print(data)
data = data.reshape(x*y,3)
compressed_data = data.copy()
#data = data.reshape(128,128,3)
#plt.imshow(data)

compression = KMeans(n_clusters = 16).fit(data)
clusterCenter = compression.cluster_centers_.copy()
labels = compression.labels_.copy()
#print(clusterCenter)
#print(labels)
for i in range(len(labels)):
    compressed_data[i] = clusterCenter[labels[i]]
compressed_data = compressed_data.reshape(x,y,3)
#print(compressed_data)
plt.imshow(compressed_data)






























