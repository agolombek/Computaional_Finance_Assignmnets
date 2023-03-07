# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:25:25 2023

@author: arong
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


x = np.linspace(-2,2,10**3)
b_range = np.logspace(-2, 0, 5)
a_range = np.linspace(1, 10, 5)


for b in b_range:
    cdf = norm.cdf(x, loc=0, scale=b)
    lab = r'$\sigma_s = $'+f'{round(b,2)}'
    plt.plot(x, cdf, label=lab)

plt.xlabel(r'$x$',fontsize=16)
plt.ylabel(r'$f(x)$',fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.grid()
# plt.savefig('smoothing_cdf.pdf', format="pdf")
plt.show()

for a in a_range:
    sigmoid = 1/(1+np.exp(-a*x))
    lab = r'$a = $'+f'{round(a,2)}'
    plt.plot(x,sigmoid, label=lab)
    
plt.xlabel(r'$x$',fontsize=16)
plt.ylabel(r'$g(x)$',fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
# plt.savefig('smoothing_sigmoid.pdf', format="pdf")
plt.show()
    