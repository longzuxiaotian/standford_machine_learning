# -*- coding: utf-8 -*-
import scipy.io as scio
from PIL import Image

data = scio.loadmat("ex3data1.mat")
#print(data['y'])
'''
print(numpy.append(0,[1,2,3]))
y = (1==2)
print(y)
'''


X = data['X'][600].reshape((20,20)) * 255
newL = Image.fromarray(X).convert('L')
newL.save('code.jpg','jpeg')


'''
image = Image.new('L',(100,100))
image.save('code.jpg','jpeg')
'''