# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:58:03 2023

@author: arong
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd

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
    
    
    
sigma = 0.2
S0 = 100
T = 1
N = 50
K = 99
r = 0.06

seed = 0

for i in range(50):
    t, S = Black_Scholes_Euler(S0, r, N, T, sigma, seed)
    seed += 1
    plt.plot(t, S)

plt.xlabel('time')
plt.ylabel(r'$S_{t}$')
plt.grid()
plt.show()
    
    
    
    
    