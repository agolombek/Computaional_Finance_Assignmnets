# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:58:03 2023

@author: arong
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import math
from time import time

@njit
def Black_Scholes_Euler(S0, r, N, T, sigma, seed):
    
    np.random.seed(seed)
    
    dt = T/N
    t_values = np.linspace(0, T, N+1)
    
    S = S0
    S_values = np.zeros(N+1)
    S_values[0] = S
    
    for i in np.arange(1, N+1):
        ds = r*S*dt + sigma*S*np.sqrt(dt)*np.random.normal()
        S = S + ds
        S_values[i] = S 
    return t_values, S_values


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


def Hedge_Simulation(S_array, K, r, T, t_array, sigma, num_rebalances):
    
    # Find intervals at which to reblance delta hedge and convert to indicies
    N = len(t_array)
    rebalance_step = math.floor(N/(num_rebalances+1))
    rebalance_idxs = [int(i*rebalance_step) for i in range(1, num_rebalances + 1)]
    
    # Sell Option for price dictated by Black Scholes Equation
    S0 = S_array[0]
    account = BlackScholesAnalytical(S0, K, r, T, sigma)
    
    
    # Perform initial delta hedge
    tau = T
    d1 = (np.log(S0/K) + (r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    current_delta = norm.cdf(d1)
    account += -S0*current_delta*np.exp(r*tau)

    
    for idx in rebalance_idxs:
        
        # tau = time to maturity
        tau = T - t_array[idx] 
        # S = current stock price
        S = S_array[idx] 
        # d1 is paramter determined by Black-Scholes Equation
        d1 = (np.log(S/K) + (r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
        # new_delta is the resulting delta hedge to be obtained
        new_delta = norm.cdf(d1)
        
        # update acount as follows:
        # if more delta is required that money is borrowed from bank at
        # interest rate for remaining time i.e e^r*tau
        # if less delta is required sell stock and invest money at interest 
        # rate for remaining time period e^r*tau
        account += -(new_delta-current_delta)*S*np.exp(r*tau)

        current_delta = new_delta
    
    # sell delta of stock at 1final price
    account += current_delta*S_array[-1]
    
    account += -np.maximum(S_array[-1]-K, 0)
    
    return account
    
true_sigma = 0.2
S0 = 100
T = 1
N = 10001
K = 99
r = 0.06

seed = 0

# num_rebalances = 4
# t_array, S_array = Black_Scholes_Euler(S0, r, N, T, sigma, seed)
# profit = Hedge_Simulation(S_array, K, r, T, t_array, sigma, num_rebalances)

iterations = 100
sigmas = [0.01, 0.1, 0.2, 0.3, 0.4]

rebalances = np.logspace(1, 3, 100)
start = time()
colors = ['red', 'orange', 'black', 'cornflowerblue','green']
col_idx = 0

for sigma in sigmas:
    profit_means = []
    profit_std = []
    for num_rebalances in rebalances:
        profits = []
        for i in range(iterations):
            t_array, S_array = Black_Scholes_Euler(S0, r, N, T, true_sigma, seed)
            profit = Hedge_Simulation(S_array, K, r, T, t_array, sigma, int(num_rebalances))
            profits.append(profit)
            
            seed += 1
        
        profit_means.append(np.mean(profits))
        profit_std.append(np.std(profits))
    
    profit_means = np.array(profit_means)
    profit_std = np.array(profit_std)
    
    sigma_diff = round(true_sigma - sigma, 2)
    plot_label = r'$\Delta \sigma =$' + f'{sigma_diff}'
    plt.plot(rebalances, profit_means, color=colors[col_idx], label=plot_label)
    plt.fill_between(rebalances, profit_means + profit_std, profit_means - profit_std, color=colors[col_idx], alpha = 0.5) 
    
    col_idx += 1
    
plt.legend(fontsize=9, loc='lower right')  
plt.grid("both")
plt.ylabel("Profit", fontsize=12)
plt.xlabel("Number of Rebalances", fontsize=12)
plt.xscale('log')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('hedging_simulation_plot_trial_1.pdf', format="pdf")
plt.show()

end = time()
print("elapsed time = ", (end-start)/60, " minutes")

    
    
    