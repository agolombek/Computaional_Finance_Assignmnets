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

################### Black Scholes - Binomial Tree Comparisson #################

sigma = 0.2
S = 100
T = 1
N = 50
K = 99
r = 0.06

tree = buildTree(S, sigma, T, N)

option_type = "call"
matrix = EuropeanOptionValueMatrix(tree, T, r, K, sigma, option_type)
print("Call Option Value at t=0 evaluated using binomial tree with 50 steps = ", matrix[0][0])
black_scholes_value = BlackScholesAnalytical(S, K, r, T, sigma)
print("Call Option Value at t=0 evaluated using analytical Black Scholes Equation = ",black_scholes_value)

tree = buildTree(S, sigma, T, N)
option_type = "put"
matrix = EuropeanOptionValueMatrix(tree, T, r, K, sigma, option_type)
print("Put Option Value at t=0 evaluated using binomial tree with 50 steps = ", matrix[0][0])

########################### Studying Convergence ##############################

"""
Studying the Convergence of a European Call Option for increasing number of
steps (N) and compared against the analystical solution given by the closed
form solution of the Black Scholes Equations.
"""

def convergence_analysis(N_low, N_high, S, T, K, r, sigma, option_type):
    
    all_N = np.arange(N_low, N_high+1)
    values = np.zeros(np.shape(all_N))
    i = 0
    
    for N in all_N:
        
        tree = buildTree(S, sigma, T, N)
        value = EuropeanOptionValueMatrix(tree, T, r, K, sigma, option_type)[0][0]
    
        values[i] = value
        i += 1
    
    return all_N, values
    

# N_low = 1
# N_high = 10000

# # Call Option
# all_N, values = convergence_analysis(N_low, N_high, S, T, K, r, sigma, "call")

# dictionary = {"N values": all_N, "Option Valuation": values, "Black Scholes": black_scholes_value}
# df = pd.DataFrame(dictionary)
# df.to_csv("./European_Option_Results/eur_call_varying_N.csv", index=False)

# # Put Option
# all_N, values = convergence_analysis(N_low, N_high, S, T, K, r, sigma, "put")

# dictionary = {"N values": all_N, "Option Valuation": values}
# df = pd.DataFrame(dictionary)
# df.to_csv("./European_Option_Results/eur_put_varying_N.csv", index=False)

# print("Completed European Convergence Analysis")


######################### Hedge Parameter Analysis ############################

# Caluclate Hedge Parameter for different Volatilities and compare to Black Scholes

def BlackScholesHedge(S, K, r, T, vol):
    """
    Hedge parameter determined by Black Scholes Equation
    """
    d1 = (np.log(S/K) + (r+0.5*vol**2)*T)/(vol*np.sqrt(T))
    N_d1 = norm.cdf(d1)
    return N_d1


def Hedge_t0_BinomialTree(S, N, T, K, r, sigma, option_type):
    
    tree = buildTree(S, sigma, T, N)
    tree_copy = copy(tree)
    valuematrix = EuropeanOptionValueMatrix(tree, T, r, K, sigma, option_type)
    delta_hedge = (valuematrix[1][1] - valuematrix[1][0])/(tree_copy[1][1] - tree_copy[1][0])
    return delta_hedge
    

sigma = 0.2
S = 100
T = 1
N = 50
K = 99
r = 0.06

option_type = "call"
print("Call Option Delta at t=0 evaluated using binomial tree with 50 steps = ", Hedge_t0_BinomialTree(S, N, T, K, r, sigma, option_type))
print("Call Option Delta at t=0 evaluated using analytical Black Scholes Equation = ", BlackScholesHedge(S, K, r, T, sigma))
    

# Compare Heding Paramter of both methods over differen volatilities

volatilities = np.linspace(0.01, 6, 1000)
BS_hedge = []
BT_hedge = []

for sigma in volatilities:
    BT_hedge.append(Hedge_t0_BinomialTree(S, N, T, K, r, sigma, option_type))
    BS_hedge.append(BlackScholesHedge(S, K, r, T, sigma))

plt.plot(volatilities, BT_hedge, color='b', label='Binomial Tree')
plt.plot(volatilities, BS_hedge, color='black', label='Black Scholes', linestyle='dashed')
plt.xlabel(r"$\sigma$", fontsize=16)
plt.ylabel(r'$\Delta_{0}$', fontsize=16)
plt.legend(fontsize=14)
plt.grid()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('./European_Option_Results/call_option_hedge_varying_sigma.pdf', format="pdf")
plt.show()


abs_diff = np.absolute(np.array(BT_hedge)-np.array(BS_hedge))

plt.plot(volatilities, abs_diff, color='b')
plt.xlabel(r"$\sigma$", fontsize=16)
plt.ylabel('Absolute Difference', fontsize=16)
plt.grid()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('./European_Option_Results/call_option_hedge_varying_sigma_diff.pdf', format="pdf")
plt.show()  

# create zoomed plots
start = 0
stop = 90

plt.plot(volatilities[start:stop], BT_hedge[start:stop], color='b', label='Binomial Tree')
plt.plot(volatilities[start:stop], BS_hedge[start:stop], color='black', label='Black Scholes', linestyle='dashed')
plt.xlabel(r"$\sigma$", fontsize=16)
plt.ylabel(r'$\Delta_{0}$', fontsize=16)
plt.legend(fontsize=14)
plt.grid()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('./European_Option_Results/call_option_hedge_varying_sigma_zoom.pdf', format="pdf")
plt.show()


abs_diff = np.absolute(np.array(BT_hedge)-np.array(BS_hedge))

plt.plot(volatilities[start:stop], abs_diff[start:stop], color='b')
plt.xlabel(r"$\sigma$", fontsize=16)
plt.ylabel('Absolute Difference', fontsize=16)
plt.grid()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('./European_Option_Results/call_option_hedge_varying_sigma_diff_zoom.pdf', format="pdf")
plt.show()  
    







