# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:25:50 2023

@author: arong
"""

import numpy as np
from numba import njit
from time import time
from scipy.stats import norm
import matplotlib.pyplot as plt


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

################### Black Scholes - Bonimial Tree Comparisson #################

sigma = 0.2
S = 100
T = 1
N = 50
K = 99
r = 0.06

tree = buildTree(S, sigma, T, N)
matrix = valueOptionMatrix(tree, T, r, K, sigma)
print("Option Value at t=0 evaluated using binomial tree with 50 steps = ", matrix[0][0])
black_scholes_value = BlackScholesAnalytical(S, K, r, T, sigma)
print("Option Value at t=0 evaluated using analytical Black Scholes Equation = ",black_scholes_value)

########################### Studying Convergence ##############################


def convergence_analysis(N_low, N_high, S, T, K, r, sigma):
    
    all_N = np.arange(N_low, N_high+1)
    values = np.zeros(np.shape(all_N))
    i = 0
    
    for N in all_N:
        
        tree = buildTree(S, sigma, T, N)
        value = valueOptionMatrix(tree, T, r, K, sigma)[0][0]
    
        values[i] = value
        i += 1
    
    return all_N, values
    

def PlotOptionValue(all_N, values, black_scholes_value):
    # Plot Initial Option value as a function of N
    low_N = all_N[0]
    high_N = all_N[-1]
    
    plt.plot(all_N, values, 'b', label = "Binomial Tree")
    plt.plot(all_N, np.ones(len(all_N))*black_scholes_value, 'r', label = "Analytical Black Scholes Solution")
    plt.legend()
    plt.grid("both")
    plt.ylabel("Option Value at t=0")
    plt.xlabel("N")
    plt.tight_layout()
    plt.savefig(f'./European_Option_Results/option_value_{low_N}N_to_{high_N}N.pdf', format="pdf")
    plt.show()
    

def PlotAbsoluteDifference(all_N, values, black_scholes_value):
    # Plot abolute value of difference between Binomial Tree valuation and 
    # analytical valuation of Black Scholes Model
    low_N = all_N[0]
    high_N = all_N[-1]
    
    absolute_difference = np.abs(values-black_scholes_value)
    plt.plot(all_N, absolute_difference, color='green')
    plt.grid("both")
    plt.ylabel("Absolute Difference")
    plt.xlabel("N")
    plt.tight_layout()
    plt.savefig(f'./European_Option_Results/absolute_difference_{low_N}N_to_{high_N}N.pdf', format="pdf")
    plt.show()


N_low = 1
N_high = 1000

all_N, values = convergence_analysis(N_low, N_high, S, T, K, r, sigma)

# Plot all values of N
PlotOptionValue(all_N, values,black_scholes_value)
PlotAbsoluteDifference(all_N, values, black_scholes_value)

# Create graphs of certain ranges
start = 10
stop = 100 
PlotOptionValue(all_N[start:stop], values[start:stop], black_scholes_value)
PlotAbsoluteDifference(all_N[start:stop], values[start:stop], black_scholes_value)

start = 100
stop = 1000 
PlotOptionValue(all_N[start:stop], values[start:stop], black_scholes_value)
PlotAbsoluteDifference(all_N[start:stop], values[start:stop], black_scholes_value)





    


    







