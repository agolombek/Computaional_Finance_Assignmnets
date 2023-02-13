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

def PlotOptionValue(all_N, values, option_type):
    # Plot Initial Option value as a function of N
    low_N = all_N[0]
    high_N = all_N[-1]
    
    plt.plot(all_N, values, 'green', label = "European Put Option")
    plt.legend()
    plt.grid("both")
    plt.ylabel("Option Value at t=0")
    plt.xlabel("N")
    plt.tight_layout()
    plt.savefig(f'./{option_type}/option_value_{low_N}N_to_{high_N}N.pdf', format="pdf")
    plt.show()
    


    
df = pd.read_csv('eur_put_varying_N.csv',delimiter=",")
all_N = np.array(df["N values"])
values = np.array(df["Option Valuation"])


# Plot all values of N
PlotOptionValue(all_N, values, "put")


# Create graphs of certain ranges
start = 10
stop = 100 
PlotOptionValue(all_N[start:stop], values[start:stop], "put")


start = 100
stop = 1000
PlotOptionValue(all_N[start:stop], values[start:stop], "put")


start = 900
stop = 1000
PlotOptionValue(all_N[start:stop], values[start:stop], "put")
