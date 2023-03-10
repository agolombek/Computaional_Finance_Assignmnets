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
def EvaluateAsianOption(S_matrix, N, r, T, n_sim, K):
    
    geo_values = np.zeros(n_sim)
    arit_values = np.zeros(n_sim)
    i = 0
    
    for S_values in S_matrix:
        
        geo_average = np.exp(np.log(S_values).mean())
        geo_payoff = np.maximum(geo_average-K, 0)
        geo_value = geo_payoff*np.exp(-r*T)
        geo_values[i] = geo_value
        
        arit_average = np.mean(S_values)
        arit_payoff = np.maximum(arit_average-K, 0)
        arit_value = arit_payoff*np.exp(-r*T)
        arit_values[i] = arit_value
        
        i += 1
    
    return geo_values, arit_values

@njit
def Control_Variates(geo_values, arit_values, n_sim, analytical):
    
    cov_matrix = np.cov(np.stack((geo_values, arit_values)))
    
    geo_mean = np.mean(geo_values)
    geo_var = cov_matrix[0][0]
    geo_SE = np.sqrt(geo_var/n_sim)
    
    arit_mean = np.mean(arit_values)
    arit_var = cov_matrix[1][1]
    arit_SE = np.sqrt(arit_var/n_sim)
    
    co_var = cov_matrix[0][1]
    
    c = -co_var/geo_var
    
    z_mean = arit_mean + c*(geo_mean - analytical)
    z_var = arit_var + np.square(c)*geo_var + 2*c*co_var
    z_SE = np.sqrt(z_var/n_sim)
    
    z_values = arit_values + c*(geo_values-analytical)
    z_mean1 = np.mean(z_values)
    # print(z_mean, z_mean1)
    # print(z_var, np.square(np.std(z_values)))
    
    return geo_mean, arit_mean, z_mean, geo_SE, arit_SE, z_SE
    
    
def AsianOptionValueAnalytical(S0, r, T, sigma, N, K):
    
    sigma_tilde =  sigma*np.sqrt((2*N+1)/(6*(N+1)))
    r_tilde = (r - 0.5*np.square(sigma) + np.square(sigma_tilde))/2
    
    d1_tilde = (np.log(S0/K) + T*(r_tilde+0.5*np.square(sigma_tilde)))/(sigma_tilde*np.sqrt(T))
    d2_tilde = (np.log(S0/K) + T*(r_tilde-0.5*np.square(sigma_tilde)))/(sigma_tilde*np.sqrt(T))
    
    value = np.exp(-r*T)*(S0*np.exp(r_tilde*T)*norm.cdf(d1_tilde)-K*norm.cdf(d2_tilde))
    
    return value
    

######################## Varying number of Paths #############################

sigma = 0.2
S0 = 100
K = 99
T = 1
r = 0.06

N = 10**4

analytical = AsianOptionValueAnalytical(S0, r, T, sigma, N, K)

num_sims = np.logspace(1, 4, 50)

geo_means = np.zeros(len(num_sims))
geo_SEs =  np.zeros(len(num_sims))

arit_means = np.zeros(len(num_sims))
arit_SEs =  np.zeros(len(num_sims))

Z_means = np.zeros(len(num_sims))
Z_SEs =  np.zeros(len(num_sims))

seed = 0
i = 0

for n_sim in num_sims:
    
    n_sim = int(n_sim)
    np.random.seed(seed)
    
    Z_matrix = np.random.normal(size=(n_sim, N))
    
    S_matrix = Black_Scholes(S0, r, N, T, sigma, seed, n_sim, Z_matrix)
    
    geo_values, arit_values = EvaluateAsianOption(S_matrix, N, r, T, n_sim, K)
    

    geo_mean, arit_mean, z_mean, geo_SE, arit_SE, z_SE = Control_Variates(geo_values, arit_values, n_sim, analytical)
    
    geo_means[i] = geo_mean
    geo_SEs[i] = geo_SE

    arit_means[i] = arit_mean
    arit_SEs[i] = arit_SE

    Z_means[i] = z_mean
    Z_SEs[i] = z_SE
    
    seed += 1
    i += 1

plt.plot(num_sims, analytical*np.ones(len(num_sims)), color='black', label='Analytical', linestyle='dashed')
plt.plot(num_sims, geo_means, color='C0', label='Monte Carlo - Geometric Average')
plt.fill_between(num_sims, geo_means + geo_SEs, geo_means - geo_SEs, color='C0', alpha = 0.2) 
plt.xlabel('Number of Simulations',fontsize=16)
plt.ylabel(r'$V_{0}$',fontsize=16)
plt.xscale('log')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('geometric_monte_carlo_convergence.pdf', format="pdf")
plt.show()

plt.plot(num_sims, geo_means, color='C0', label='Geometric Average')
plt.fill_between(num_sims, geo_means + geo_SEs, geo_means - geo_SEs, color='C0', alpha = 0.2) 
plt.plot(num_sims, arit_means, color='green', label='Arithmetic Average')
plt.fill_between(num_sims, arit_means + arit_SEs, arit_means - arit_SEs, color='green', alpha = 0.2) 
plt.plot(num_sims, Z_means, color='red', label='Arithmetiv Avergae with CVs')
plt.fill_between(num_sims, Z_means + Z_SEs, Z_means - Z_SEs, color='red', alpha = 0.2) 
plt.xlabel('Number of Simulations',fontsize=16)
plt.ylabel(r'$V_{0}$',fontsize=16)
plt.xscale('log')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('control_variates_pricing_varying_nsim.pdf', format="pdf")
plt.show()

plt.plot(num_sims, geo_SEs, color='C0', label='Geometric Average')
plt.plot(num_sims, arit_SEs, color='green', label='Arithmetic Average')
plt.plot(num_sims, Z_SEs, color='red', label='Arithmetiv Avergae with CVs')
plt.xlabel('Number of Simulations',fontsize=16)
plt.ylabel('Standard Error',fontsize=16)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=14, loc='upper right')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('control_variates_SE_varying_nsim.pdf', format="pdf")
plt.show()

###################### Varying number of Time Points ##########################

sigma = 0.2
S0 = 100
K = 99
T = 1
r = 0.06

n_sim = 10**3
N_range = np.logspace(1,5,50)


analytical = np.zeros(len(N_range))

geo_means = np.zeros(len(N_range))
geo_SEs =  np.zeros(len(N_range))

arit_means = np.zeros(len(N_range))
arit_SEs =  np.zeros(len(N_range))

Z_means = np.zeros(len(N_range))
Z_SEs =  np.zeros(len(N_range))

seed = 0
i = 0

for N in N_range:

    N = int(N)
    np.random.seed(seed)
    
    Z_matrix = np.random.normal(size=(n_sim, N))
    
    S_matrix = Black_Scholes(S0, r, N, T, sigma, seed, n_sim, Z_matrix)
    
    analytical[i] = AsianOptionValueAnalytical(S0, r, T, sigma, N, K)
    
    geo_values, arit_values = EvaluateAsianOption(S_matrix, N, r, T, n_sim, K)
    
    geo_mean, arit_mean, z_mean, geo_SE, arit_SE, z_SE = Control_Variates(geo_values, arit_values, n_sim, analytical[i])
    
    geo_means[i] = geo_mean
    geo_SEs[i] = geo_SE

    arit_means[i] = arit_mean
    arit_SEs[i] = arit_SE

    Z_means[i] = z_mean
    Z_SEs[i] = z_SE
    
    seed += 1
    i += 1

plt.plot(N_range, analytical, color='black', label='Analytical', linestyle='dashed')
plt.plot(N_range, geo_means, color='C0', label='Monte Carlo - Geometric Average')
plt.fill_between(N_range, geo_means + geo_SEs, geo_means - geo_SEs, color='C0', alpha = 0.2) 
plt.xlabel('Number of Time Points',fontsize=16)
plt.ylabel(r'$V_{0}$',fontsize=16)
plt.xscale('log')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('geometric_monte_carlo_convergence_timesteps.pdf', format="pdf")
plt.show()

plt.plot(N_range, geo_means, color='C0', label='Geometric Average')
plt.fill_between(N_range, geo_means + geo_SEs, geo_means - geo_SEs, color='C0', alpha = 0.2) 
plt.plot(N_range, arit_means, color='green', label='Arithmetic Average')
plt.fill_between(N_range, arit_means + arit_SEs, arit_means - arit_SEs, color='green', alpha = 0.2) 
plt.plot(N_range, Z_means, color='red', label='Arithmetiv Avergae with CVs')
plt.fill_between(N_range, Z_means + Z_SEs, Z_means - Z_SEs, color='red', alpha = 0.2) 
plt.xlabel('Number of Time Points',fontsize=16)
plt.ylabel(r'$V_{0}$',fontsize=16)
plt.xscale('log')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('control_variates_pricing_varying_timesteps.pdf', format="pdf")
plt.show()

plt.plot(N_range, geo_SEs, color='C0', label='Geometric Average')
plt.plot(N_range, arit_SEs, color='green', label='Arithmetic Average')
plt.plot(N_range, Z_SEs, color='red', label='Arithmetiv Avergae with CVs')
plt.xlabel('Number of Time Points',fontsize=16)
plt.ylabel('Standard Error',fontsize=16)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('control_variates_SE_varying_timesteps.pdf', format="pdf")
plt.show()

######################### Varying Strike Prices #############################

sigma = 0.2
S0 = 100
K = 99
T = 1
r = 0.06
n_sim = 10**3
N = 10**2

K_range = np.linspace(50, 140, 91)

analytical = np.zeros(len(K_range))

geo_means = np.zeros(len(K_range))
geo_SEs =  np.zeros(len(K_range))

arit_means = np.zeros(len(K_range))
arit_SEs =  np.zeros(len(K_range))

Z_means = np.zeros(len(K_range))
Z_SEs =  np.zeros(len(K_range))

seed = 0
i = 0

for K in K_range:

    K = int(K)
    np.random.seed(seed)
    
    Z_matrix = np.random.normal(size=(n_sim, N))
    
    S_matrix = Black_Scholes(S0, r, N, T, sigma, seed, n_sim, Z_matrix)
    
    analytical[i] = AsianOptionValueAnalytical(S0, r, T, sigma, N, K)
    
    geo_values, arit_values = EvaluateAsianOption(S_matrix, N, r, T, n_sim, K)
    
    geo_mean, arit_mean, z_mean, geo_SE, arit_SE, z_SE = Control_Variates(geo_values, arit_values, n_sim, analytical[i])
    
    geo_means[i] = geo_mean
    geo_SEs[i] = geo_SE

    arit_means[i] = arit_mean
    arit_SEs[i] = arit_SE

    Z_means[i] = z_mean
    Z_SEs[i] = z_SE
    
    seed += 1
    i += 1

plt.plot(K_range, analytical, color='black', label='Analytical', linestyle='dashed')
plt.plot(K_range, geo_means, color='C0', label='Monte Carlo - Geometric Average')
plt.fill_between(K_range, geo_means + geo_SEs, geo_means - geo_SEs, color='C0', alpha = 0.2) 
plt.xlabel('K',fontsize=16)
plt.ylabel(r'$V_{0}$',fontsize=16)
# plt.xscale('log')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('geometric_monte_carlo_convergence_K.pdf', format="pdf")
plt.show()

plt.plot(K_range, geo_means, color='C0', label='Geometric Average')
plt.fill_between(K_range, geo_means + geo_SEs, geo_means - geo_SEs, color='C0', alpha = 0.2) 
plt.plot(K_range, arit_means, color='green', label='Arithmetic Average')
plt.fill_between(K_range, arit_means + arit_SEs, arit_means - arit_SEs, color='green', alpha = 0.2) 
plt.plot(K_range, Z_means, color='red', label='Arithmetiv Avergae with CVs')
plt.fill_between(K_range, Z_means + Z_SEs, Z_means - Z_SEs, color='red', alpha = 0.2) 
plt.xlabel('K',fontsize=16)
plt.ylabel(r'$V_{0}$',fontsize=16)
plt.xscale('log')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('control_variates_pricing_varying_K.pdf', format="pdf")
plt.show()


plt.plot(K_range, geo_SEs, color='C0', label='Geometric Average')
plt.plot(K_range, arit_SEs, color='green', label='Arithmetic Average')
plt.plot(K_range, Z_SEs, color='red', label='Arithmetiv Avergae with CVs')
plt.xlabel('Ks',fontsize=16)
plt.ylabel('Standard Error',fontsize=16)
# plt.xscale('log')
# plt.yscale('log')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('control_variates_SE_varying_K.pdf', format="pdf")
plt.show()

######################### Varying Volatility ################################

# sigma = 0.2
S0 = 100
K = 99
K = 99
T = 1
r = 0.06
n_sim = 10**3
N = 10**2

volatilities = np.linspace(0.01, 1, 50)

analytical = np.zeros(len(volatilities))

geo_means = np.zeros(len(volatilities))
geo_SEs =  np.zeros(len(volatilities))

arit_means = np.zeros(len(volatilities))
arit_SEs =  np.zeros(len(volatilities))

Z_means = np.zeros(len(volatilities))
Z_SEs =  np.zeros(len(volatilities))

seed = 0
i = 0

for sigma in volatilities:

    np.random.seed(seed)
    
    Z_matrix = np.random.normal(size=(n_sim, N))
    
    S_matrix = Black_Scholes(S0, r, N, T, sigma, seed, n_sim, Z_matrix)
    
    analytical[i] = AsianOptionValueAnalytical(S0, r, T, sigma, N, K)
    
    geo_values, arit_values = EvaluateAsianOption(S_matrix, N, r, T, n_sim, K)
    
    geo_mean, arit_mean, z_mean, geo_SE, arit_SE, z_SE = Control_Variates(geo_values, arit_values, n_sim, analytical[i])
    
    geo_means[i] = geo_mean
    geo_SEs[i] = geo_SE

    arit_means[i] = arit_mean
    arit_SEs[i] = arit_SE

    Z_means[i] = z_mean
    Z_SEs[i] = z_SE
    
    seed += 1
    i += 1

plt.plot(volatilities, analytical, color='black', label='Analytical', linestyle='dashed')
plt.plot(volatilities, geo_means, color='C0', label='Monte Carlo - Geometric Average')
plt.fill_between(volatilities, geo_means + geo_SEs, geo_means - geo_SEs, color='C0', alpha = 0.2) 
plt.xlabel(r'$\sigma$',fontsize=16)
plt.ylabel(r'$V_{0}$',fontsize=16)
# plt.xscale('log')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('geometric_monte_carlo_convergence_sigma.pdf', format="pdf")
plt.show()

plt.plot(volatilities, geo_means, color='C0', label='Geometric Average')
plt.fill_between(volatilities, geo_means + geo_SEs, geo_means - geo_SEs, color='C0', alpha = 0.2) 
plt.plot(volatilities, arit_means, color='green', label='Arithmetic Average')
plt.fill_between(volatilities, arit_means + arit_SEs, arit_means - arit_SEs, color='green', alpha = 0.2) 
plt.plot(volatilities, Z_means, color='red', label='Arithmetiv Avergae with CVs')
plt.fill_between(volatilities, Z_means + Z_SEs, Z_means - Z_SEs, color='red', alpha = 0.2) 
plt.xlabel(r'$\sigma$',fontsize=16)
plt.ylabel(r'$V_{0}$',fontsize=16)
# plt.xscale('log')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('control_variates_pricing_varying_sigma.pdf', format="pdf")
plt.show()


plt.plot(volatilities, geo_SEs, color='C0', label='Geometric Average')
plt.plot(volatilities, arit_SEs, color='green', label='Arithmetic Average')
plt.plot(volatilities, Z_SEs, color='red', label='Arithmetiv Avergae with CVs')
plt.xlabel(r'$\sigma$',fontsize=16)
plt.ylabel('Standard Error',fontsize=16)
# plt.xscale('log')
# plt.yscale('log')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('control_variates_SE_varying_sigma.pdf', format="pdf")
plt.show()