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

    return option_value


def AnalyticalDigitalDelta(S, K, r, T, vol):
    """
    https://quant.stackexchange.com/questions/23267/delta-of-binary-option
    """
    d1 = (np.log(S/K) + (r+0.5*vol**2)*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    N_d2 = norm.pdf(d2)
    
    delta_0 = np.exp(-r*T)*N_d2/(S*sigma*np.sqrt(T))
    return delta_0


def STD_FDM(set_1, set_2, bump, n_sim):
    
    cov_matrix = np.cov(np.stack((set_1, set_2), axis=0))
    
    var_1 = cov_matrix[0][0]
    var_2 = cov_matrix[1][1]

    cov = cov_matrix[0][1]
    
    total_SE = np.sqrt((var_1 + var_2 - 2*cov)/np.square(bump))/np.sqrt(n_sim)
    
    return total_SE


def STD_CDM(set_1, set_2, bump, n_sim):
    
    cov_matrix = np.cov(np.stack((set_1, set_2), axis=0))
    
    var_1 = cov_matrix[0][0]
    var_2 = cov_matrix[1][1]
    
    cov = cov_matrix[0][1]
    
    total_SE = np.sqrt((var_1 + var_2 - 2*cov)/np.square(2*bump))/np.sqrt(n_sim)
    
    return total_SE


sigma = 0.2
S0 = 100
T = 1
K = 99
r = 0.06
n_sim = 10**5


#################### Bump and Revalue - Using Same Seed #######################

bumps = np.linspace(0.0001, 0.1, 1000)

analytical = AnalyticalDigitalDelta(S0, K, r, T, sigma)*np.ones(len(bumps))

seed = 0
np.random.seed(seed)
Z  = np.random.normal(size=n_sim)
Z1  = np.random.normal(size=n_sim)
base_value = DigitalMonteCarlo(S0, r, T, K, sigma, n_sim, Z)
base_avg = np.mean(base_value)

FDM_delta = np.zeros(len(bumps))
FDM_std = np.zeros(len(bumps))

CDM_delta = np.zeros(len(bumps))
CDM_std = np.zeros(len(bumps))


i = 0
for h in bumps:
    
    S_f = S0 + S0*h
    S_b = S0 - S0*h
    
    bumped_values = DigitalMonteCarlo(S_f, r, T, K, sigma, n_sim, Z1)
    bump_avg = np.mean(bumped_values)
    
    backward_bump_values= DigitalMonteCarlo(S_b, r, T, K, sigma, n_sim, Z)
    bk_bump_avg = np.mean(backward_bump_values)
    
    
    delta_f = (bump_avg-base_avg)/(h*S0)
    FDM_delta[i] = delta_f
    FDM_std[i] = STD_FDM(base_value, bumped_values, S0*h, n_sim)
    
    delta_c = (bump_avg-bk_bump_avg)/(2*h*S0)
    CDM_delta[i] = delta_c
    CDM_std[i] = STD_CDM(bumped_values, backward_bump_values, S0*h, n_sim)
    
    i+=1

plt.plot(bumps*100, analytical, color='black', label='Analytical', linestyle='dashed')

plt.plot(bumps*100, FDM_delta, color='C0', label='Forward Difference')
plt.fill_between(bumps*100, FDM_delta + FDM_std, FDM_delta - FDM_std, color='C0', alpha = 0.5)

plt.plot(bumps*100, CDM_delta, color='green', label='Central Difference')
plt.fill_between(bumps*100, CDM_delta + CDM_std, CDM_delta - CDM_std, color='green', alpha = 0.5)

plt.xlabel('h [%]',fontsize=16)
plt.ylabel(r'$\Delta_{0}$',fontsize=16)
plt.legend(fontsize=14)
plt.xscale('log')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig('digital_bump_delta_diff_seed.pdf', format="pdf")
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
plt.savefig('digital_bump_SE_diff_seed.pdf', format="pdf")
plt.show()

####################### PathwiseDerivative Estimate ###########################

@njit
def DigitalPathiwseDeltaSigmoid(S0, r, T, K, sigma, n_sim, Z, a):
    """
    Using Monte Carlo Approach to find the delta hedge parameter of a
    digital option using a sigmoid as a smoothing function.
    """
    ST = S0*np.exp(T*(r-0.5*np.square(sigma)) + sigma*Z*np.sqrt(T))
    
    x = ST - K 
    dVdST = (a*np.exp(-a*x))/np.square(np.exp(-a*x)+1)
    
    deltas = dVdST*np.exp(-r*T)*ST/S0
    avg_val = np.mean(deltas)
    std_val = np.std(deltas)
    
    return avg_val, std_val

smoothing_values_sigmoid = np.logspace(-1, 1, 1000)

analytical = AnalyticalDigitalDelta(S0, K, r, T, sigma)*np.ones(len(smoothing_values_sigmoid))

Pathwise_sigmoid_Delta = np.zeros(len(smoothing_values_sigmoid))
pathwise_sigmoid_std = np.zeros(len(smoothing_values_sigmoid))

i = 0
 
for a in smoothing_values_sigmoid:
    
    delta, std_val = DigitalPathiwseDeltaSigmoid(S0, r, T, K, sigma, n_sim, Z, a)
    Pathwise_sigmoid_Delta[i] = delta
    pathwise_sigmoid_std[i] = std_val/np.sqrt(n_sim)
    i += 1
    


def DigitalPathiwseDeltaCDF(S0, r, T, K, sigma, n_sim, Z, b):
    """
    Using Monte Carlo Approach to find the delta hedge parameter of a
    digital option using the cumulitive density dunction of a normal 
    distribution as a smoothing function.
    """
    ST = S0*np.exp(T*(r-0.5*np.square(sigma)) + sigma*Z*np.sqrt(T))

    x = ST - K

    dVdST = norm.pdf(x, loc=0, scale=b)
    
    deltas = dVdST*np.exp(-r*T)*ST/S0
    avg_val = np.mean(deltas)
    std_val = np.std(deltas)
    
    return avg_val, std_val

smoothing_values_cdf = np.logspace(-2, 1, 1000)

Pathwise_cdf_Delta = np.zeros(len(smoothing_values_cdf))
pathwise_cdf_std = np.zeros(len(smoothing_values_cdf))

i = 0
 
for b in smoothing_values_cdf:
    delta, std_val = DigitalPathiwseDeltaCDF(S0, r, T, K, sigma, n_sim, Z, b)
    Pathwise_cdf_Delta[i] = delta
    pathwise_cdf_std[i] = std_val/np.sqrt(n_sim)
    i += 1



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
std = std*np.ones(len(smoothing_values_sigmoid))/np.sqrt(n_sim)

################################ Plot ########################################

plt.plot(smoothing_values_sigmoid, analytical, color='black', label='Analytical', linestyle='dashed')
plt.plot(smoothing_values_sigmoid, likelihood, color='blue', label='Likelihood ratio')
plt.fill_between(smoothing_values_sigmoid, likelihood + std, likelihood - std, color='blue', alpha = 0.2)
plt.plot(smoothing_values_sigmoid, Pathwise_sigmoid_Delta, color='red', label='Pathwise Method Sigmoid')
plt.fill_between(smoothing_values_sigmoid, Pathwise_sigmoid_Delta + pathwise_sigmoid_std, Pathwise_sigmoid_Delta - pathwise_sigmoid_std, color='red', alpha = 0.2)
plt.xscale('log')
plt.xlabel(r'$a$',fontsize=16)
plt.ylabel(r'$\Delta_0$',fontsize=16)
plt.legend(fontsize=14, loc = 'lower right')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.grid()
# plt.savefig('digital_option_delta_sigmoid.pdf', format="pdf")
plt.show()

plt.plot(smoothing_values_cdf, analytical, color='black', label='Analytical', linestyle='dashed')
plt.plot(smoothing_values_cdf, likelihood, color='blue', label='Likelihood ratio')
plt.fill_between(smoothing_values_cdf, likelihood + std, likelihood - std, color='blue', alpha = 0.2)
plt.plot(smoothing_values_cdf, Pathwise_cdf_Delta, color='green', label='Pathwise Method Normal CDF')
plt.fill_between(smoothing_values_cdf, Pathwise_cdf_Delta + pathwise_cdf_std, Pathwise_cdf_Delta - pathwise_cdf_std, color='green', alpha = 0.2)
plt.xscale('log')
plt.gca().invert_xaxis()
plt.xlabel(r'$\sigma_s$',fontsize=16)
plt.ylabel(r'$\Delta_0$',fontsize=16)
plt.legend(fontsize=14, loc = 'lower right')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.grid()
# plt.savefig('digital_option_delta_cdf.pdf', format="pdf")
plt.show()

plt.plot(smoothing_values_sigmoid, std, color='blue', label='Likelihood ratio')
plt.plot(smoothing_values_sigmoid, pathwise_sigmoid_std, color='red', label='Pathwise Method Sigmoid')
plt.xscale('log')
plt.xlabel(r'$a$',fontsize=16)
plt.ylabel('Standard Error',fontsize=16)
plt.legend(fontsize=14, loc = 'upper left')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.grid()
# plt.savefig('digital_option_delta_sigmoid_SE.pdf', format="pdf")
plt.show()

plt.plot(smoothing_values_sigmoid, std, color='blue', label='Likelihood ratio')
plt.plot(smoothing_values_sigmoid, pathwise_cdf_std, color='green', label='Pathwise Method Normal CDF')
plt.xscale('log')
plt.gca().invert_xaxis()
plt.xlabel(r'$\sigma_s$',fontsize=16)
plt.ylabel('Standard Error',fontsize=16)
plt.legend(fontsize=14, loc = 'upper left')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.grid()
# plt.savefig('digital_option_delta_cdf_SE.pdf', format="pdf")
plt.show()