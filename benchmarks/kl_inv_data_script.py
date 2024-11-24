# creates a csv file with entries that solve the kl inverse problem for any fraction that is equal 
#   to x / 1000

import numpy as np
import cvxpy as cp
import pandas as pd
import os
from lah.utils.nn_utils import (
    invert_kl,
)


# def solve_kl_inv(q, pen):
#     # solve the kl inverse problem
#     p_star = invert_kl
#     return p_star

N = 1000
delta = 1e-5
pen = np.log(2 / delta) / N

kl_invs = np.zeros(N + 1)
for i in range(N + 1):
    emp_risk = i / N
    kl_inv = invert_kl(emp_risk, pen)
    # print('kl_inv', kl_inv)
    kl_invs[i] = kl_inv

df = pd.DataFrame()
df['kl_inv'] = kl_invs
os.mkdir('kl_inv_cache')
df.to_csv(f"kl_inv_cache/kl_inv_delta_{delta}_Nval_{N}.csv")
