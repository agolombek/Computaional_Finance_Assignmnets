# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:23:41 2023

@author: arong
"""

import numpy as np
from numba import njit
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import diags, SparseEfficiencyWarning
from scipy.stats import norm


def FTCS_matrix(T, N, S_min, S_max, M, K, r, sigma):
    """
    T = maturity time
    N = number of time steps
    S_max = boundary condition for S
    M = discretization in space
    r = risk free interest rate
    sigma = volatility
    """
    grid = np.zeros((M, N))
    
    X_min = np.log(S_min)
    X_max = np.log(S_max)

    dx = (X_max-X_min)/M
    dt = T/N 
    
    time = np.linspace(0, T, N)
    X = np.linspace(X_min, X_max, M)
 
    # set boundary conditions
    # top boundary S = 0 so option price = 0
    grid[0][:] = 0
    
    # bottom boundary i.e. X_max option value approaches S_max
    grid[-1][:] = S_max
    
    # at tau = 0 or t = T option price is max(S-K, 0)
    x_boundary = np.exp(X[1:M-1])
    x_boundary = np.maximum(x_boundary-K, 0)

    for j in range(1,M-1):
        grid[j,0] = x_boundary[j-1]
    
    # create A matrix
    a_minus_1 = (0.5*dt/dx)*(np.square(sigma)/dx-(r-0.5*np.square(sigma)))
    a0 = 1-np.square(sigma)*dt/np.square(dx)-r*dt
    a1 = (0.5*dt/dx)*(np.square(sigma)/dx+(r-0.5*np.square(sigma)))
    
    diag_minus_1 = np.ones(M-1)*a_minus_1
    diag_minus_1[-1] = 0
    
    diag_0 = np.ones(M)*a0
    diag_0[0] = 0
    diag_0[-1] = 0
    
    diag_1 = np.ones(M-1)*a1
    diag_1[0] = 0
    
    diagonals = [diag_minus_1, diag_0, diag_1]
    A = diags(diagonals, [-1, 0, 1])
    
    k = np.zeros(M)
    k[-1] = S_max
    
    # iterate over columns
    for i in range(1, N):
        vn = grid[:,i-1]
        vn1 = A.dot(vn) + k
        grid[:,i] = vn1


    return time, X, grid
    

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
    
T = 1 
r = 0.04
sigma = 0.3
S0 = 100
K = 110

S_min = 10**(-4)
S_max = 10**4

N = 10**3
M = 10**3

time, X, grid = FTCS_matrix(T, N, S_min, S_max, M, K, r, sigma)

############################# Compare to Analytical Solution ##################

# extract final column i.e. tau = T, t = 0 from grid
V0 = grid[:,-1]

# extract tested X values and transform to S
S0_FTCS = np.exp(X)

# find analytical solution for corresponding S0 values
analytical = []
for S in S0_FTCS:
    analytical.append(BlackScholesAnalytical(S, K, r, T, sigma))


# plot FTCS and analytical solution
plt.plot(S0_FTCS, V0, label='FTCS')
plt.plot(S0_FTCS, analytical, label='BS')
plt.legend()
plt.grid()
plt.xlabel('S0')
plt.ylabel('V0')
plt.show()

# plot difference
plt.plot(S0_FTCS, V0-np.array(analytical))
plt.grid()
plt.xlabel('S0')
plt.ylabel('Difference')
plt.show()

# interpolarte FTCS array to find exact approximation at S0
print(f"FTCS solution for S0 = {S0}: ", np.interp(S0, S0_FTCS, V0))
print(f"analyticalsolution for S0 = {S0}: ", BlackScholesAnalytical(S0, K, r, T, sigma))

######################## 3D Plot #############################################

time = np.flip(time)
X, Y = np.meshgrid(time, np.exp(X))
Z = grid
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('time [years]', fontsize=12, rotation=150)
ax.set_ylabel('S', fontsize=12)
ax.set_zlabel('V', fontsize=12, rotation=60)
plt.show()
