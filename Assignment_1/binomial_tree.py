# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:25:50 2023

@author: arong
"""

import numpy as np
from numba import njit

@njit
def buildTree(S, vol, T, N):
    dt = T/N
    
    matrix = np.zeros((N+1, N+1))
    
    u = np.exp(vol*np.sqrt(dt))
    d = np.exp(-vol*np.sqrt(dt))
    
    for i in np.arange(N+1):
        for j in np.arange(i+1):
            matrix[i,j] = S * u**j * d**(i-j)
    
    return matrix

sigma = 0.1
S = 80
T = 1
N = 2

tree = buildTree(S, sigma, T, N)
