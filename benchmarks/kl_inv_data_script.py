# creates a csv file with entries that solve the kl inverse problem for any fraction that is equal 
#   to x / 1000

import numpy as np
import cvxpy as cp
import pandas as pd


def solve_kl_inv(q, pen):
    # solve the kl inverse problem
    p_star = 0
    return p_star

N = 1000
delta = 1e-5
pen = np.log(2 / delta) / N

kl_invs = np.zeros(N)
for i in range(N + 1):
    emp_risk = i / N
    kl_inv = solve_kl_inv(emp_risk, pen)
    kl_invs[i] = kl_inv

df = pd.DataFrame()
df['kl_invs'] = kl_invs
df.to_csv('kl_inv_csv')
