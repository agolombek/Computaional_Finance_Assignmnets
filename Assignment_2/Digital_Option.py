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
    
    return avg_val, std_val, np.mean(maturity_price)


def AnalyticalDigitalDelta(S, K, r, T, vol):
    """
    https://quant.stackexchange.com/questions/23267/delta-of-binary-option
    """
    d1 = (np.log(S/K) + (r+0.5*vol**2)*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    N_d2 = norm.pdf(d2)
    
    delta_0 = np.exp(-r*T)*N_d2/(S*sigma*np.sqrt(T))
    return delta_0

sigma = 0.2
S0 = 100
T = 1
K = 99
r = 0.06
n_sim = 10**5


#################### Bump and Revalue - Using Same Seed #######################

bumps = np.linspace(0.0000001, 0.5, 1000)

analytical = AnalyticalDigitalDelta(S0, K, r, T, sigma)*np.ones(len(bumps))

seed = 0
np.random.seed(seed)
Z  = np.random.normal(size=n_sim)
base_value, std, avg_ST = DigitalMonteCarlo(S0, r, T, K, sigma, n_sim, Z)

FDM_delta = np.zeros(len(bumps))
CDM_delta = np.zeros(len(bumps))

i = 0
for h in bumps:
    
    S_f = S0 + S0*h
    S_b = S0 - S0*h

    bumped_value, std, avg_ST = DigitalMonteCarlo(S_f, r, T, K, sigma, n_sim, Z)
    backward_bump_value, std, avg_ST = DigitalMonteCarlo(S_b, r, T, K, sigma, n_sim, Z)
    
    delta_f = (bumped_value-base_value)/(h*S0)
    FDM_delta[i] = delta_f
    
    delta_c = (bumped_value-backward_bump_value)/(2*h*S0)
    CDM_delta[i] = delta_c
    i+=1

plt.plot(bumps*100, analytical, color='black', label='Analytical', linestyle='dashed')
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

####################### PathwiseDerivative Estimate ###########################

@njit
def DigitalPathiwseDeltaSigmoid(S0, r, T, K, sigma, n_sim, Z, a):
    """
    Using Monte Carlo Approach to find the delta hedge parameter of a
    digital option using a sigmoid as a smoothing function.
    """
    ST = S0*np.exp(T*(r-0.5*np.square(sigma)) + sigma*Z*np.sqrt(T))
    
    e_term = np.exp(-a*(ST-K))

    dVdST = (a*e_term)/np.square(e_term+1)
    
    deltas = dVdST*np.exp(-r*T)*ST/S0
    avg_val = np.mean(deltas)
    std_val = np.std(deltas)
    
    return avg_val, std_val

smoothing_values_sigmoid = np.linspace(0.1, 10, 100)
analytical = AnalyticalDigitalDelta(S0, K, r, T, sigma)*np.ones(len(smoothing_values_sigmoid))

Pathwise_sigmoid_Delta = np.zeros(len(smoothing_values_sigmoid))
pathwise_sigmoid_std = np.zeros(len(smoothing_values_sigmoid))

i = 0
 
for a in smoothing_values_sigmoid:
    delta, std_val = DigitalPathiwseDeltaSigmoid(S0, r, T, K, sigma, n_sim, Z, a)
    Pathwise_sigmoid_Delta[i] = delta
    pathwise_sigmoid_std[i] = 1.96*std_val/np.sqrt(n_sim)
    i += 1


# @njit
def DigitalPathiwseDeltaCDF(S0, r, T, K, sigma, n_sim, Z, b):
    """
    Using Monte Carlo Approach to find the delta hedge parameter of a
    digital option using the cumulitive density dunction of a normal 
    distribution as a smoothing function.
    """
    ST = S0*np.exp(T*(r-0.5*np.square(sigma)) + sigma*Z*np.sqrt(T))

    dVdST = norm.pdf(ST, loc=K, scale=b)
    
    deltas = dVdST*np.exp(-r*T)*ST/S0
    avg_val = np.mean(deltas)
    print(avg_val)
    std_val = np.std(deltas)
    
    return avg_val, std_val

smoothing_values_cdf = np.linspace(0.01, 2, 100)

Pathwise_cdf_Delta = np.zeros(len(smoothing_values_cdf))
pathwise_cdf_std = np.zeros(len(smoothing_values_cdf))

i = 0
 
for b in smoothing_values_cdf:
    delta, std_val = DigitalPathiwseDeltaCDF(S0, r, T, K, sigma, n_sim, Z, b)
    Pathwise_cdf_Delta[i] = delta
    pathwise_cdf_std[i] = 1.96*std_val/np.sqrt(n_sim)
    i += 1

plt.plot(smoothing_values_cdf, analytical, color='black', label='Analytical', linestyle='dashed')
plt.plot(smoothing_values_cdf, Pathwise_cdf_Delta, color='green', label='Pathwise Method')
plt.fill_between(smoothing_values_cdf, Pathwise_cdf_Delta + pathwise_cdf_std, Pathwise_cdf_Delta - pathwise_cdf_std, color='green', alpha = 0.2) 
plt.show()

####################### LikelihoodRatioDerivative Estimate ###########################

@njit
def DigitalLikelihoodDelta(S0, r, T, K, sigma, Z):
    """
    Using Monte Carlo Approach to find the delta hedge parameter of a
    digital option - Likelihood ratio.
    """
    ST = S0*np.exp(T*(r-0.5*np.square(sigma)) + sigma*Z*np.sqrt(T))

    payoff = ST > K
    
    deltas = np.exp(-r * T) * payoff * Z / (S0 * sigma * np.sqrt(T))
    avg_val = np.mean(deltas)
    std_val = np.std(deltas)
    
    return avg_val, std_val
    
likelihood, std = DigitalLikelihoodDelta(S0, r, T, K, sigma, Z)

likelihood = likelihood*np.ones(len(smoothing_values_sigmoid))
std = 1.96*std*np.ones(len(smoothing_values_sigmoid))/np.sqrt(n_sim)

plt.plot(smoothing_values_sigmoid, analytical, color='black', label='Analytical', linestyle='dashed')

plt.plot(smoothing_values_sigmoid, Pathwise_sigmoid_Delta, color='red', label='Pathwise Method')
plt.fill_between(smoothing_values_sigmoid, Pathwise_sigmoid_Delta + pathwise_sigmoid_std, Pathwise_sigmoid_Delta - pathwise_sigmoid_std, color='red', alpha = 0.2) 

plt.plot(smoothing_values_sigmoid, likelihood, color='blue', label='Likelihood ratio')
plt.fill_between(smoothing_values_sigmoid, likelihood + std, likelihood - std, color='blue', alpha = 0.2) 

plt.xlabel('a',fontsize=12)
plt.ylabel(r'$\Delta_{0}$',fontsize=12)
plt.legend()
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
plt.grid()
plt.tight_layout()
# plt.savefig('digital_option_delta.pdf', format="pdf")
plt.show()
