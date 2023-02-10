# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:25:50 2023

@author: arong
"""

import numpy as np
from numba import njit
from time import time

@njit
def buildTree(S, vol, T, N):
    """
    Build binomial tree containing possible stock prices at discrete time 
    steps dt. Each row in the output matrix advances the time.
    
    S = Stock price at t=0
    vol = volatility
    T = maturity time
    N = number of time steps taken until T
    """
    dt = T/N
    
    matrix = np.zeros((N+1, N+1))
    
    u = np.exp(vol*np.sqrt(dt))
    d = np.exp(-vol*np.sqrt(dt))
    
    for i in np.arange(N+1):
        for j in np.arange(i+1):
            matrix[i,j] = S * u**j * d**(i-j)
    
    return matrix

@njit
def valueOptionMatrix(tree, T, r, K, vol):
    """
    Given an binomial tree with stock prices at every dt, iterate backwards
    through the tree to calculate the value of the option at every point.
    """
    
    columns = tree.shape[1]
    rows = tree.shape[0]
    
    dt = T/(rows-1)

    u = np.exp(vol*np.sqrt(dt))
    d = np.exp(-vol*np.sqrt(dt))
    
    p = (np.exp(r*dt)-d)/(u-d)
    
    # Calculate Payoff of option in the last row
    for c in np.arange(columns):
        S = tree[rows-1, c]
        tree[rows-1, c] = np.maximum(S-K, 0)
    
    # Iterate backwards through the tree
    for i in np.arange(rows-1)[::-1]:
        for j in np.arange(i+1):
            down = tree[i+1, j]
            up = tree[i+1, j+1]
            tree[i, j] = np.exp(-r*dt)*(p*up + (1-p)*down)
    
    return tree


############################## TEST CASE ######################################

# sigma = 0.1
# S = 80
# T = 1
# N = 2
# K = 85
# r = 0.1

# tree = buildTree(S, sigma, T, N)
# print(tree)
# matrix = valueOptionMatrix(tree, T, r, K, sigma)
# print(tree)

############################## Experiment #####################################

sigma = 0.2
S = 100
T = 1
N = 5000
K = 99
r = 0.06

start = time()

tree = buildTree(S, sigma, T, N)
matrix = valueOptionMatrix(tree, T, r, K, sigma)
print(tree[0][0])

end = time()
print("Runtime = ", end- start, " seconds")






