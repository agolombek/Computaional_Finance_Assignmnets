# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:19:23 2023

@author: arong
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

@njit
def Black_Scholes(S0, r, N, T, sigma, seed, n_sim, Z_matrix):
    
    t_values = np.linspace(0, T, N)
    S_matrix = np.zeros(shape=(n_sim, N))
    
    dt = T/N
    
    for j in np.arange(n_sim):
        S = S0
        S_matrix[j][0] = S
        for i in np.arange(1, N):
            Z = Z_matrix[j][i]
            S =  S*np.exp(dt*(r-0.5*np.square(sigma)) + sigma*Z*np.sqrt(dt))
            S_matrix[j][i] = S
        
    return S_matrix

@njit
def EvaluateAsianOption(S_matrix, N, r, T, n_sim):
    
    values = np.zeros(n_sim)
    i = 0
    
    for S_values in S_matrix:
        geometric_average = np.exp(np.log(S_values).mean())
        payoff = np.maximum(geometric_average-K, 0)
        value = payoff*np.exp(-r*T)
        
        values[i] = value
        i += 1
    
    return values

def AsianOptionValueAnalytical(S0, r, T, sigma, N):
    
    sigma_tilde =  sigma*np.sqrt((2*N+1)/(6*(N+1)))
    r_tilde = (r - 0.5*np.square(sigma) + np.square(sigma_tilde))/2
    
    d1_tilde = (np.log(S0/K) + T*(r_tilde+0.5*np.square(sigma_tilde)))/(sigma_tilde*np.sqrt(T))
    d2_tilde = (np.log(S0/K) + T*(r_tilde-0.5*np.square(sigma_tilde)))/(sigma_tilde*np.sqrt(T))
    
    value = np.exp(-r*T)*(S0*np.exp(r_tilde*T)*norm.cdf(d1_tilde)-K*norm.cdf(d2_tilde))
    
    return value
    
sigma = 0.2
S0 = 100
K = 99
T = 1
r = 0.06

N = 10**4

analytical = AsianOptionValueAnalytical(S0, r, T, sigma, N)

########## Test convergence for different number of simulations ###############

num_sims = np.logspace(1, 4, 50)

mean = np.zeros(len(num_sims))
SE =  np.zeros(len(num_sims))

seed = 0
i = 0

for n_sim in num_sims:
    
    n_sim = int(n_sim)
    np.random.seed(seed)
    
    Z_matrix = np.random.normal(size=(n_sim, N))
    
    S_matrix = Black_Scholes(S0, r, N, T, sigma, seed, n_sim, Z_matrix)
    
    values = EvaluateAsianOption(S_matrix, N, r, T, n_sim)
    
    mean[i] = np.mean(values)
    SE[i] = np.std(values)/np.sqrt(n_sim)
    
    seed += 1
    i += 1

plt.plot(num_sims, analytical*np.ones(len(num_sims)), color='black', label='Analytical', linestyle='dashed')
plt.plot(num_sims, mean, color='C0', label='Monte Carlo')
plt.fill_between(num_sims, mean + SE, mean - SE, color='C0', alpha = 0.5) 
plt.xlabel('Number of Simulations',fontsize=12)
plt.ylabel(r'$V_{0}$',fontsize=12)
plt.xscale('log')
plt.legend()
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
plt.grid()
plt.tight_layout()
# plt.savefig('asian_call_option_varying_nsim.pdf', format="pdf")
plt.show()

# Test convergence using different values for time steps
