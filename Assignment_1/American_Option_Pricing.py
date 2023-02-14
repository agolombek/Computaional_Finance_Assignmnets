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
from copy import copy
from European_Option_Pricing import EuropeanOptionValueMatrix, buildTree


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


##################### Values for Comparisson to European ######################
sigma = 0.2
S = 100
T = 1
N = 50
K = 99
r = 0.06

option_type = "call"
tree = buildTree(S, sigma, T, N)
matrix = AmericanOptionValueMatrix(tree, T, r, K, sigma, option_type)
print("Call Option Value at t=0 evaluated using binomial tree with 50 steps = ", matrix[0][0])

option_type = "put"
tree = buildTree(S, sigma, T, N)
matrix = AmericanOptionValueMatrix(tree, T, r, K, sigma, option_type)
print("Put Option Value at t=0 evaluated using binomial tree with 50 steps = ", matrix[0][0])

########################### Studying Convergence ##############################

def convergence_analysis(N_low, N_high, S, T, K, r, sigma, option_type):
    
    all_N = np.arange(N_low, N_high+1)
    values = np.zeros(np.shape(all_N))
    i = 0
    
    for N in all_N:
        
        tree = buildTree(S, sigma, T, N)
        value = AmericanOptionValueMatrix(tree, T, r, K, sigma, option_type)[0][0]
    
        values[i] = value
        i += 1
    
    return all_N, values
    

# N_low = 1
# N_high = 1000

# # Call Option
# all_N, values = convergence_analysis(N_low, N_high, S, T, K, r, sigma, "call")

# dictionary = {"N values": all_N, "Option Valuation": values}
# df = pd.DataFrame(dictionary)
# df.to_csv("./American_Option_Results/ame_call_varying_N.csv", index=False)

# # Put Option
# all_N, values = convergence_analysis(N_low, N_high, S, T, K, r, sigma, "put")

# dictionary = {"N values": all_N, "Option Valuation": values}
# df = pd.DataFrame(dictionary)
# df.to_csv("./American_Option_Results/ame_put_varying_N.csv", index=False)

########################### Studying Volatility ##############################

S = 100
T = 1
N = 50
K = 99
r = 0.06

volatilities = np.linspace(0.01, 10, 1000)
ame_call = []
ame_put = []
eur_call = []
eur_put = []

for sigma in volatilities:
    tree = buildTree(S, sigma, T, N)
    tree1 = copy(tree)
    tree2 = copy(tree)
    tree3 = copy(tree)
    
    option_type = "call"
    ame_call.append(AmericanOptionValueMatrix(tree, T, r, K, sigma, option_type)[0][0])
    eur_call.append(EuropeanOptionValueMatrix(tree1, T, r, K, sigma, option_type)[0][0])
    
    option_type = "put"
    ame_put.append(AmericanOptionValueMatrix(tree2, T, r, K, sigma, option_type)[0][0])
    eur_put.append(EuropeanOptionValueMatrix(tree3, T, r, K, sigma, option_type)[0][0])


plt.plot(volatilities, ame_call, color='red', label='American Call Option')
plt.plot(volatilities, eur_call, color='b', label='European Call Option', linestyle='dashed')
plt.plot(volatilities, ame_put, color='orange', label='American Put Option')
plt.plot(volatilities, eur_put, color='green', label='European Put Option', linestyle='dashed')
plt.xlabel(r"$\sigma$")
plt.ylabel('Option Value at t= 0')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('./American_Option_Results/option_value_varying_sigma.pdf', format="pdf")
plt.show()    