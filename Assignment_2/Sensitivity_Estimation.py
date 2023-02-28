# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:13:41 2023

@author: arong
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


@njit
def monte_carlo_european_put(S0, r, T, K, sigma, n_sim, Z):
    """
    Using Monte Carlo Approach to price a european put option.
    """
    maturity_price = S0*np.exp(T*(r-0.5*np.square(sigma)) + sigma*Z*np.sqrt(T))
    
    maturity_payoff = np.maximum(K - maturity_price, np.zeros(n_sim))
    
    option_value = maturity_payoff*np.exp(-r*T)

    avg_val = np.mean(option_value)
    std_val = np.std(option_value)
    
    return avg_val, std_val

def BlackScholesPutHedge(S, K, r, T, vol):
    """
    Caluclates the anayltical solution for delta hedge of a european put 
    option.
    """
    d1 = (np.log(S/K) + (r+0.5*vol**2)*T)/(vol*np.sqrt(T))
    hedge = -norm.cdf(-d1)

    return hedge

    
sigma = 0.2
S0 = 100
T = 1
K = 99
r = 0.06
n_sim = 10**5

# Percentage by which S0 will be bumped
bumps = np.linspace(0.01, 0.2, 50)

BS_delta = BlackScholesPutHedge(S0, K, r, T, sigma)*np.ones(len(bumps))

seed = 0
np.random.seed(seed)
Z  = np.random.normal(size=n_sim)
base_value, std = monte_carlo_european_put(S0, r, T, K, sigma, n_sim, Z)

FDM_delta = np.zeros(len(bumps))

i = 0
for h in bumps:
    
    seed += 1
    seed = 0
    S0 = S0 + S0*h
    np.random.seed(seed)
    Z  = np.random.normal(size=n_sim)
    bumped_value, std = monte_carlo_european_put(S0, r, T, K, sigma, n_sim, Z)
    
    delta = (bumped_value-base_value)/(h*S0)
    FDM_delta[i] = delta
    i+=1

plt.plot(bumps, BS_delta, color='black', label='Analytical', linestyle='dashed')
plt.plot(bumps, FDM_delta, color='C0', label='Finite Difference')
plt.xlabel('h [%]',fontsize=12)
plt.ylabel(r'$\Delta_{0}$',fontsize=12)
plt.legend()
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
plt.grid()
plt.tight_layout()
# plt.savefig('european_put_option_varying_nsim.pdf', format="pdf")
plt.show()