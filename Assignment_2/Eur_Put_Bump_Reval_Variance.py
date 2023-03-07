# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:11:33 2023

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

    return option_value

def BlackScholesPutHedge(S, K, r, T, vol):
    """
    Caluclates the anayltical solution for delta hedge of a european put 
    option.
    """
    d1 = (np.log(S/K) + (r+0.5*vol**2)*T)/(vol*np.sqrt(T))
    hedge = -norm.cdf(-d1)

    return hedge


def STD_FDM(set_1, set_2, bump, n_sim):
    
    cov_matrix = np.cov(np.stack((set_1, set_2), axis=0))
    
    var_1 = cov_matrix[0][0]
    var_2 = cov_matrix[1][1]

    cov = cov_matrix[0][1]
    
    total_SE = np.sqrt((var_1 + var_2 + 2*cov)/np.square(bump))/np.sqrt(n_sim)
    
    return total_SE


def STD_CDM(set_1, set_2, bump, n_sim):
    
    cov_matrix = np.cov(np.stack((set_1, set_2), axis=0))
    
    var_1 = cov_matrix[0][0]
    var_2 = cov_matrix[1][1]
    
    cov = cov_matrix[0][1]
    
    total_SE = np.sqrt((var_1 + var_2 + 2*cov)/np.square(2*bump))/np.sqrt(n_sim)
    
    return total_SE

    
sigma = 0.2
S0 = 100
T = 1
K = 99
r = 0.06
n_sim = 10**5

# Percentage by which S0 will be bumped

bumps = np.linspace(0.0001, 0.1, 1000)

BS_delta = BlackScholesPutHedge(S0, K, r, T, sigma)*np.ones(len(bumps))

#################### Bump and Revalue - Using Same Seed #######################

seed = 0
np.random.seed(seed)
Z  = np.random.normal(size=n_sim)
base_values = monte_carlo_european_put(S0, r, T, K, sigma, n_sim, Z)
base_avg = np.mean(base_values)

FDM_delta = np.zeros(len(bumps))
FDM_std = np.zeros(len(bumps))

CDM_delta = np.zeros(len(bumps))
CDM_std = np.zeros(len(bumps))

i = 0
for h in bumps:
    
    S_f = S0 + S0*h
    S_b = S0 - S0*h

    bumped_values = monte_carlo_european_put(S_f, r, T, K, sigma, n_sim, Z)
    bump_avg = np.mean(bumped_values)
    
    backward_bump_values= monte_carlo_european_put(S_b, r, T, K, sigma, n_sim, Z)
    bk_bump_avg = np.mean(backward_bump_values)
    
    
    delta_f = (bump_avg-base_avg)/(h*S0)
    FDM_delta[i] = delta_f
    FDM_std[i] = STD_FDM(base_values, bumped_values, S0*h, n_sim)
    
    delta_c = (bump_avg-bk_bump_avg)/(2*h*S0)
    CDM_delta[i] = delta_c
    CDM_std[i] = STD_CDM(bumped_values, backward_bump_values, S0*h, n_sim)
    
    i+=1

plt.plot(bumps*100, BS_delta, color='black', label='Analytical', linestyle='dashed')

plt.plot(bumps*100, FDM_delta, color='C0', label='Forward Difference')
# plt.fill_between(bumps*100, FDM_delta + FDM_std, FDM_delta - FDM_std, color='C0', alpha = 0.5)

plt.plot(bumps*100, CDM_delta, color='green', label='Central Difference')
# plt.fill_between(bumps*100, CDM_delta + CDM_std, CDM_delta - CDM_std, color='green', alpha = 0.5)

plt.xlabel('h [%]',fontsize=16)
plt.ylabel(r'$\Delta_{0}$',fontsize=16)
plt.legend(fontsize=14)
plt.xscale('log')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('european_put_option_delta_same_seed.pdf', format="pdf")
plt.show()

plt.plot(bumps*100, FDM_std, color='C0', label='Forward Difference')
plt.plot(bumps*100, CDM_std, color='green', label='Central Difference')
plt.xlabel('h [%]',fontsize=16)
plt.ylabel('Standrad Error',fontsize=16)
plt.legend(fontsize=14)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig('european_put_option_SE_same_seed.pdf', format="pdf")
plt.show()



################# Bump and Revalue - Using Different Seed #####################

seed = 0
np.random.seed(seed)
Z  = np.random.normal(size=n_sim)
base_values = monte_carlo_european_put(S0, r, T, K, sigma, n_sim, Z)
base_avg = np.mean(base_values)

FDM_delta = np.zeros(len(bumps))
FDM_std = np.zeros(len(bumps))

CDM_delta = np.zeros(len(bumps))
CDM_std = np.zeros(len(bumps))

i = 0
for h in bumps:
    
    seed += 1
    np.random.seed(seed)
    Z  = np.random.normal(size=n_sim)
    
    S_f = S0 + S0*h
    S_b = S0 - S0*h

    bumped_values = monte_carlo_european_put(S_f, r, T, K, sigma, n_sim, Z)
    bump_avg = np.mean(bumped_values)
    
    backward_bump_values= monte_carlo_european_put(S_b, r, T, K, sigma, n_sim, Z)
    bk_bump_avg = np.mean(backward_bump_values)
    
    
    delta_f = (bump_avg-base_avg)/(h*S0)
    FDM_delta[i] = delta_f
    FDM_std[i] = STD_FDM(base_values, bumped_values, S0*h, n_sim)
    
    delta_c = (bump_avg-bk_bump_avg)/(2*h*S0)
    CDM_delta[i] = delta_c
    CDM_std[i] = STD_CDM(bumped_values, backward_bump_values, S0*h, n_sim)
    
    i+=1

plt.plot(bumps*100, BS_delta, color='black', label='Analytical', linestyle='dashed')

plt.plot(bumps*100, FDM_delta, color='C0', label='Forward Difference')
# plt.fill_between(bumps*100, FDM_delta + FDM_std, FDM_delta - FDM_std, color='C0', alpha = 0.5)

plt.plot(bumps*100, CDM_delta, color='green', label='Central Difference')
# plt.fill_between(bumps*100, CDM_delta + CDM_std, CDM_delta - CDM_std, color='green', alpha = 0.5)

plt.xlabel('h [%]',fontsize=16)
plt.ylabel(r'$\Delta_{0}$',fontsize=16)
plt.legend(fontsize=14)
plt.xscale('log')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('european_put_option_delta_diff_seed.pdf', format="pdf")
plt.show()

plt.plot(bumps*100, FDM_std, color='C0', label='Forward Difference')
plt.plot(bumps*100, CDM_std, color='green', label='Central Difference')
plt.xlabel('h [%]',fontsize=16)
plt.ylabel('Standrad Error',fontsize=16)
plt.legend(fontsize=14)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig('european_put_option_SE_diff_seed.pdf', format="pdf")
plt.show()

# Create zoomed picture

start = 100
stop = len(bumps) - 1

plt.plot(bumps[start:stop]*100, BS_delta[start:stop], color='black', label='Analytical', linestyle='dashed')
plt.plot(bumps[start:stop]*100, FDM_delta[start:stop], color='C0', label='Forward Difference')
plt.plot(bumps[start:stop]*100, CDM_delta[start:stop], color='green', label='Central Difference')
plt.xlabel('h [%]',fontsize=16)
plt.ylabel(r'$\Delta_{0}$',fontsize=16)
plt.legend(fontsize=14)
plt.xscale('log')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('european_put_option_delta_diff_seed_zoomFD.pdf', format="pdf")
plt.show()

plt.plot(bumps[start:stop]*100, FDM_std[start:stop], color='C0', label='Forward Difference')
plt.plot(bumps[start:stop]*100, CDM_std[start:stop], color='green', label='Central Difference')
plt.xlabel('h [%]',fontsize=16)
plt.ylabel('Standrad Error',fontsize=16)
plt.legend(fontsize=14)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig('european_put_option_SE_diff_seed_zoom_FD.pdf', format="pdf")
plt.show()