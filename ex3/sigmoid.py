# -*- coding: utf-8 -*-
import numpy as np

def sigmoid(z) :
    return 1.0 / (1.0 + np.e ** (-1*z))
