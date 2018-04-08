# -*- coding: utf-8 -*-

import numpy as np

def math_solution(X, y):
    theta = (X.T * X) ** -1 * X.T * y
    return theta