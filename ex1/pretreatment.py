# -*- coding: utf-8 -*-

import numpy as np

def pretreatment(data):
    m = len(data)
    middle = np.mean(data, axis = 0)
    sigma = np.std(data, axis = 0)
    X = (data - np.tile(middle,(m,1))) / np.tile(sigma, (m,1))
    return [X,middle,sigma]