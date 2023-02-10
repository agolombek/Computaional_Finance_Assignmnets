# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:07:10 2023

@author: arong
"""

import numpy as np
from numba import njit
from time import time
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

def PlotOptionValue(all_N, values, black_scholes_value):
    # Plot Initial Option value as a function of N
    low_N = all_N[0]
    high_N = all_N[-1]
    
    plt.plot(all_N, values, 'b', label = "Binomial Tree")
    plt.plot(all_N, np.ones(len(all_N))*black_scholes_value, 'r', label = "Analytical Black Scholes Solution")
    plt.legend()
    plt.grid("both")
    plt.ylabel("Option Value at t=0")
    plt.xlabel("N")
    plt.tight_layout()
    plt.savefig(f'option_value_{low_N}N_to_{high_N}N.pdf', format="pdf")
    plt.show()
    

def PlotAbsoluteDifference(all_N, values, black_scholes_value):
    # Plot abolute value of difference between Binomial Tree valuation and 
    # analytical valuation of Black Scholes Model
    low_N = all_N[0]
    high_N = all_N[-1]
    
    absolute_difference = np.abs(values-black_scholes_value)
    plt.plot(all_N, absolute_difference, color='green')
    plt.grid("both")
    plt.ylabel("Absolute Difference")
    plt.xlabel("N")
    plt.tight_layout()
    plt.savefig(f'absolute_difference_{low_N}N_to_{high_N}N.pdf', format="pdf")
    plt.show()
    
df = pd.read_csv('european_option_evaluation.csv',delimiter=",")
all_N = np.array(df["N values"])
values = np.array(df["Option Valuation"])
black_scholes_value = np.array(df["Black Scholes"])[0]


# Plot all values of N
PlotOptionValue(all_N, values,black_scholes_value)
PlotAbsoluteDifference(all_N, values, black_scholes_value)

# Create graphs of certain ranges
start = 10
stop = 100 
PlotOptionValue(all_N[start:stop], values[start:stop], black_scholes_value)
PlotAbsoluteDifference(all_N[start:stop], values[start:stop], black_scholes_value)

start = 100
stop = 1000
PlotOptionValue(all_N[start:stop], values[start:stop], black_scholes_value)
PlotAbsoluteDifference(all_N[start:stop], values[start:stop], black_scholes_value)

start = 1000
stop = 10000
PlotOptionValue(all_N[start:stop], values[start:stop], black_scholes_value)
PlotAbsoluteDifference(all_N[start:stop], values[start:stop], black_scholes_value)