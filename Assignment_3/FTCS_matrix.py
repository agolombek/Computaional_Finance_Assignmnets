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


def FTCS_matrix(T, N, S_min, S_max, M, K, r, sigma):
    """
    T = maturity time
    N = number of time steps
    S_max = boundary condition for S
    M = discretization in space
    r = risk free interest rate
    sigma = volatility
    """
    grid = np.zeros((N, M))
    
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
    X_boundary = np.maximum(x_boundary-K, 0)

    for j in range(1,M-1):
        grid[j][0] = x_boundary[j-1]
    
    # create A matrix
    a_minus_1 = (0.5*dt/dx)*(np.square(sigma)/dx-(r-0.5*np.square(sigma)))
    a0 = 1-np.square(sigma)*dt/dx
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
    k[-1] = np.exp(X_max)
    
    # iterate over columns
    for i in range(1, N):
        vn = grid[:,i-1]
        vn1 = A.dot(vn) + k
        grid[:,i] = vn1


    return time, X, grid
    
    
    
T = 1 
r = 0.04
sigma = 0.3
S0 = 100
K = 110

S_min = 10**(-4)
S_max = 10**4

N = 10**2
M = 10**2

time, X, grid = FTCS_matrix(T, N, S_min, S_max, M, K, r, sigma)

S = np.exp(X)
time = np.flip(time)

# np.save(f"FTCS_T_{T}_r{r}_sigma{sigma}_S0_{S0}_K_{K}.npy",grid)
# grid = np.load(f"FTCS_T_{T}_r{r}_sigma{sigma}_S0_{S0}_K_{K}.npy")

X, Y = np.meshgrid(time, S)
Z = grid

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('time [years]', fontsize=12, rotation=150)
ax.set_ylabel('S', fontsize=12)
ax.set_zlabel('V', fontsize=12, rotation=60)
plt.show()
