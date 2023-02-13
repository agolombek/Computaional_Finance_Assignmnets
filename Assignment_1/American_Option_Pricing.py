# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 12:56:40 2023

@author: arong
"""

import numpy as np
from numba import njit
from time import time
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd


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
def AmericanOptionValueMatrix(tree, T, r, K, vol, option_type):
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
    
    if option_type == "put":
    
        # Calculate Payoff of option in the last row
        for c in np.arange(columns):
            S = tree[rows-1, c]
            tree[rows-1, c] = np.maximum(K-S, 0)
        
        # Iterate backwards through the tree
        for i in np.arange(rows-1)[::-1]:
            for j in np.arange(i+1):
                S = tree[i][j]
                down = tree[i+1, j]
                up = tree[i+1, j+1]
                tree[i, j] = np.maximum(K-S, np.exp(-r*dt)*(p*up + (1-p)*down))
                
    elif option_type == "call":
        
        # Calculate Payoff of option in the last row
        for c in np.arange(columns):
            S = tree[rows-1, c]
            tree[rows-1, c] = np.maximum(S-K, 0)
        
        # Iterate backwards through the tree
        for i in np.arange(rows-1)[::-1]:
            for j in np.arange(i+1):
                S = tree[i][j]
                down = tree[i+1, j]
                up = tree[i+1, j+1]
                tree[i, j] = np.maximum(S-K, np.exp(-r*dt)*(p*up + (1-p)*down))
    
    return tree


########################### Studying Convergence ##############################


# def convergence_analysis(N_low, N_high, S, T, K, r, sigma):
    
#     all_N = np.arange(N_low, N_high+1)
#     values = np.zeros(np.shape(all_N))
#     i = 0
    
#     for N in all_N:
        
#         tree = buildTree(S, sigma, T, N)
#         value = valueOptionMatrix(tree, T, r, K, sigma)[0][0]
    
#         values[i] = value
#         i += 1
    
#     return all_N, values
    

# N_low = 1
# N_high = 1000

# all_N, values = convergence_analysis(N_low, N_high, S, T, K, r, sigma)

# dictionary = {"N values": all_N, "Option Valuation": values, "Black Scholes": black_scholes_value}
# df = pd.DataFrame(dictionary)
# df.to_csv("./European_Option_Results/european_option_evaluation.csv", index=False)

########################### Studying Volatility ##############################