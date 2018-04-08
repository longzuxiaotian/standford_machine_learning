# -*- coding: utf-8 -*-

import numpy

def sigmoid(z):
    return 1.0/(1.0+numpy.e ** (-1.0*z))