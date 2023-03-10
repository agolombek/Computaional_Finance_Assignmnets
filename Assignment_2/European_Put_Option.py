# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:44:15 2023

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
    Using Monte Carlo Approach to price a european put option
    """
    maturity_price = S0*np.exp(T*(r-0.5*np.square(sigma)) + sigma*Z*np.sqrt(T))
    
    maturity_payoff = np.maximum(K - maturity_price, np.zeros(n_sim))
    
    option_value = maturity_payoff*np.exp(-r*T)

    avg_val = np.mean(option_value)
    std_val = np.std(option_value)
    
    return avg_val, std_val


def BlackScholesAnalytical(S, K, r, T, vol):
    """
    Caluclates the anayltical solution for the value of a European put option
    given by the Black Scholes Equations.
    """
    d1 = (np.log(S/K) + (r+0.5*vol**2)*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    N_d1 = norm.cdf(-d1)
    N_d2 = norm.cdf(-d2)
    V = np.exp(-r*T)*K*N_d2 - S*N_d1
    return V


###################### Varying Number of Simulations ##########################

sigma = 0.2
S0 = 100
T = 1
K = 99
r = 0.06

analytical = BlackScholesAnalytical(S0, K, r, T, sigma)

num_sims = np.logspace(1, 6, 100)
num_sims = (np.rint(num_sims)).astype(int)

seed = 0

mean = np.zeros(len(num_sims))
std = np.zeros(len(num_sims))
i = 0

for n_sim in num_sims:
    
    np.random.seed(seed)
    Z  = np.random.normal(size=n_sim)
    
    avg_val, std_val = monte_carlo_european_put(S0, r, T, K, sigma, n_sim, Z)
    
    mean[i] = avg_val
    std[i] = std_val/np.sqrt(n_sim)
    i += 1
    
    seed += 1

plt.plot(num_sims, analytical*np.ones(len(num_sims)), color='black', label='Analytical', linestyle='dashed')
plt.plot(num_sims, mean, color='C0', label='Monte Carlo')
plt.fill_between(num_sims, mean + std, mean - std, color='C0', alpha = 0.5) 
plt.xlabel('Number of Simulations',fontsize=16)
plt.ylabel(r'$V_{0}$',fontsize=16)
plt.xscale('log')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig('european_put_option_varying_nsim.pdf', format="pdf")
plt.show()

plt.plot(num_sims, mean - analytical*np.ones(len(num_sims)), color='C0', label='Monte Carlo')
plt.fill_between(num_sims, mean - analytical*np.ones(len(num_sims)) + std, mean - analytical*np.ones(len(num_sims)) - std, color='C0', alpha = 0.5) 
plt.xlabel('Number of Simulations',fontsize=16)
plt.ylabel(r'$V_{0\_MonteCarlo} - V_{0\_BlackScholes}$',fontsize=16)
plt.xscale('log')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig('european_put_option_varying_nsim_diff.pdf', format="pdf")
plt.show()


plt.plot(num_sims, std, color='C0', label='Monte Carlo')
plt.xlabel('Number of Simulations',fontsize=16)
plt.ylabel('Standard Error',fontsize=16)
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig('european_put_option_varying_nsim_SE.pdf', format="pdf")
plt.show()

########################### Varying Volatility ################################

S0 = 100
T = 1
K = 99
r = 0.06
n_sim = 10**5

volatilities = np.linspace(0.01, 1, 100)

BlackScholes = np.zeros(len(volatilities))
MonteCarlo_mean = np.zeros(len(volatilities))
MonteCarlo_std = np.zeros(len(volatilities))

seed = 0
i = 0

for sigma in volatilities:
    
    analytical = BlackScholesAnalytical(S0, K, r, T, sigma)
    
    np.random.seed(seed)
    Z  = np.random.normal(size=n_sim)
    
    avg_val, std_val = monte_carlo_european_put(S0, r, T, K, sigma, n_sim, Z)
    
    BlackScholes[i] = analytical
    MonteCarlo_mean[i] = avg_val
    MonteCarlo_std[i] = std_val/np.sqrt(n_sim)
    i += 1
    
    seed += 1

plt.plot(volatilities, BlackScholes, color='black', label='Analytical', linestyle='dashed')
plt.plot(volatilities, MonteCarlo_mean, color='C0', label='Monte Carlo')
# plt.fill_between(volatilities, MonteCarlo_mean + MonteCarlo_std, MonteCarlo_mean - MonteCarlo_std, color='C0', alpha = 0.5) 
plt.xlabel(r'$\sigma$',fontsize=16)
plt.ylabel(r'$V_{0}$',fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig('european_put_option_varying_sigma.pdf', format="pdf")
plt.show()

plt.plot(volatilities, MonteCarlo_mean - BlackScholes, color='C0', label='Monte Carlo')
plt.fill_between(volatilities, MonteCarlo_mean - BlackScholes + MonteCarlo_std, MonteCarlo_mean - BlackScholes - MonteCarlo_std, color='C0', alpha = 0.5) 
plt.xlabel(r'$\sigma$',fontsize=16)
plt.ylabel(r'$V_{0\_MonteCarlo} - V_{0\_BlackScholes}$',fontsize=16)
# plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig('european_put_option_varying_sigma_diff.pdf', format="pdf")
plt.show()

plt.plot(volatilities, MonteCarlo_std, color='C0', label='Monte Carlo')
plt.xlabel(r'$\sigma$',fontsize=16)
plt.ylabel('Standard Error',fontsize=16)
# plt.yscale('log')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig('european_put_option_varying_sigma_SE.pdf', format="pdf")
plt.show()


########################## Varying Strike Price ###############################

sigma = 0.2
S0 = 100
T = 1
r = 0.06
n_sim = 10**5

strikes = np.linspace(1, 200, 100)

BlackScholes = np.zeros(len(volatilities))
MonteCarlo_mean = np.zeros(len(volatilities))
MonteCarlo_std = np.zeros(len(volatilities))

seed = 0
i = 0

for K in strikes:
    
    analytical = BlackScholesAnalytical(S0, K, r, T, sigma)
    
    np.random.seed(seed)
    Z  = np.random.normal(size=n_sim)
    
    avg_val, std_val = monte_carlo_european_put(S0, r, T, K, sigma, n_sim, Z)
    
    BlackScholes[i] = analytical
    MonteCarlo_mean[i] = avg_val
    MonteCarlo_std[i] = std_val/np.sqrt(n_sim)
    i += 1
    
    seed += 1

plt.plot(strikes, BlackScholes, color='black', label='Analytical', linestyle='dashed')
plt.plot(strikes, MonteCarlo_mean, color='C0', label='Monte Carlo')
# plt.fill_between(strikes, MonteCarlo_mean + MonteCarlo_std, MonteCarlo_mean - MonteCarlo_std, color='C0', alpha = 0.5) 
plt.xlabel(r'$K$',fontsize=16)
plt.ylabel(r'$V_{0}$',fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig('european_put_option_varying_K.pdf', format="pdf")
plt.show()

plt.plot(strikes, MonteCarlo_mean - BlackScholes, color='C0', label='Monte Carlo')
plt.fill_between(strikes, MonteCarlo_mean - BlackScholes + MonteCarlo_std, MonteCarlo_mean - BlackScholes - MonteCarlo_std, color='C0', alpha = 0.5) 
plt.xlabel(r'$K$',fontsize=16)
plt.ylabel(r'$V_{0\_MonteCarlo} - V_{0\_BlackScholes}$',fontsize=16)
# plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig('european_put_option_varying_K_diff.pdf', format="pdf")
plt.show()

plt.plot(strikes, MonteCarlo_std, color='C0', label='Monte Carlo')
plt.xlabel(r'$K$',fontsize=16)
plt.ylabel('Standard Error',fontsize=16)
# plt.yscale('log')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig('european_put_option_varying_K_SE.pdf', format="pdf")
plt.show()
