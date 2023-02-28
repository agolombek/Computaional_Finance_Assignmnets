# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:54:09 2023

@author: arong
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


@njit
def DigitalMonteCarlo(S0, r, T, K, sigma, n_sim, Z):
    """
    Using Monte Carlo Approach to price a digital option
    """
    maturity_price = S0*np.exp(T*(r-0.5*np.square(sigma)) + sigma*Z*np.sqrt(T))
    
    maturity_payoff = np.zeros(n_sim)
    
    for i in range(n_sim):
        if maturity_price[i] > K:
            maturity_payoff[i] = 1
    
    option_value = maturity_payoff*np.exp(-r*T)

    avg_val = np.mean(option_value)
    std_val = np.std(option_value)
    
    return avg_val, std_val


sigma = 0.2
S0 = 100
T = 1
K = 99
r = 0.06
n_sim = 10**5

bumps = np.linspace(0.0000001, 0.3, 1000)

seed = 0
np.random.seed(seed)
Z  = np.random.normal(size=n_sim)
base_value, std = DigitalMonteCarlo(S0, r, T, K, sigma, n_sim, Z)

FDM_delta = np.zeros(len(bumps))
CDM_delta = np.zeros(len(bumps))

i = 0
for h in bumps:
    
    S_f = S0 + S0*h
    S_b = S0 - S0*h

    bumped_value, std = DigitalMonteCarlo(S_f, r, T, K, sigma, n_sim, Z)
    backward_bump_value, std = DigitalMonteCarlo(S_b, r, T, K, sigma, n_sim, Z)
    
    delta_f = (bumped_value-base_value)/(h*S0)
    FDM_delta[i] = delta_f
    
    delta_c = (bumped_value-backward_bump_value)/(2*h*S0)
    CDM_delta[i] = delta_c
    i+=1

plt.plot(bumps*100, FDM_delta, color='C0', label='Forward Difference')
plt.plot(bumps*100, CDM_delta, color='green', label='Central Difference')
plt.xlabel('h [%]',fontsize=12)
plt.ylabel(r'$\Delta_{0}$',fontsize=12)
plt.legend()
plt.xscale('log')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
plt.grid()
plt.tight_layout()
# plt.savefig('digital_option_delta.pdf', format="pdf")
plt.show()