# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:38:22 2023

@author: arong
"""

import numpy as np
from numba import njit
from time import time
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from copy import copy


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
def EuropeanOptionValueMatrix(tree, T, r, K, vol, option_type):
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
    
    if option_type == "call":
        # Calculate Payoff of option in the last row
        for c in np.arange(columns):
            S = tree[rows-1, c]
            tree[rows-1, c] = np.maximum(S-K, 0)
    
    elif option_type == "put":
        # Calculate Payoff of option in the last row
        for c in np.arange(columns):
            S = tree[rows-1, c]
            tree[rows-1, c] = np.maximum(K-S, 0)
    
    # Iterate backwards through the tree
    for i in np.arange(rows-1)[::-1]:
        for j in np.arange(i+1):
            down = tree[i+1, j]
            up = tree[i+1, j+1]
            tree[i, j] = np.exp(-r*dt)*(p*up + (1-p)*down)
    
    return tree


def BlackScholesAnalytical(S, K, r, T, vol):
    """
    Caluclates the anayltical solution for the value of a European call option
    given by the Black Scholes Equations.
    """
    d1 = (np.log(S/K) + (r+0.5*vol**2)*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    V = S*N_d1 - np.exp(-r*T)*K*N_d2
    return V



S = 100
T = 1
N = 50
K = 99
r = 0.06

BS_price = []
BT_price = []

volatilities = np.linspace(0.01, 6, 1000)
option_type = "call"

for sigma in volatilities:
    option_type = "call"
    tree = buildTree(S, sigma, T, N)
    matrix = EuropeanOptionValueMatrix(tree, T, r, K, sigma, option_type)
    BT_price.append(matrix[0][0])
    
    black_scholes_value = BlackScholesAnalytical(S, K, r, T, sigma)
    BS_price.append(black_scholes_value)
    

plt.plot(volatilities, BT_price, color='b', label='Binomial Tree')
plt.plot(volatilities, BS_price, color='black', label='Black Scholes', linestyle='dashed')
plt.xlabel(r"$\sigma$", fontsize=16)
plt.ylabel(r'$V_{0}$', fontsize=16)
plt.legend(fontsize=14)
plt.grid()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('./European_Option_Results/BTM_BSM_option_price_varying_sigma.pdf', format="pdf")
plt.show()


abs_diff = np.absolute(np.array(BT_price)-np.array(BS_price))

plt.plot(volatilities, abs_diff, color='b')
plt.xlabel(r"$\sigma$", fontsize=16)
plt.ylabel('Absolute Difference', fontsize=16)
plt.grid()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('./European_Option_Results/BTM_BSM_option_price_varying_sigma_diff.pdf', format="pdf")
plt.show()  
    