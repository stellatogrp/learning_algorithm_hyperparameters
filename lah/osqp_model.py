from functools import partial

import jax.numpy as jnp
import numpy as np
import osqp
from scipy.sparse import csc_matrix

from lah.algo_steps_osqp import k_steps_eval_osqp, k_steps_train_osqp, unvec_symm
from lah.l2ws_model import L2WSmodel


class OSQPmodel(L2WSmodel):
    def __init__(self, **kwargs):
        super(OSQPmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        # self.m, self.n = self.A.shape
        self.algo = 'osqp'
        self.lah = False
        self.m, self.n = input_dict['m'], input_dict['n']
        self.q_mat_train, self.q_mat_test = input_dict['q_mat_train'], input_dict['q_mat_test']

        self.rho = input_dict['rho']
        self.sigma = input_dict.get('sigma', 1)
        self.alpha = input_dict.get('alpha', 1)
        self.output_size = self.n + self.m

        # custom_loss
        custom_loss = input_dict.get('custom_loss')

        """
        break into the 2 cases
        1. factors are the same for each problem (i.e. matrices A and P don't change)
        2. factors change for each problem
        """
        self.factors_required = True
        self.factor_static_bool = input_dict.get('factor_static_bool', True)
        if self.factor_static_bool:
            self.A = input_dict['A']
            # self.P = input_dict.get('P', None)
            self.P = input_dict['P']
            self.factor_static = input_dict['factor']
            self.k_steps_train_fn = partial(
                k_steps_train_osqp, A=self.A, rho=self.rho, sigma=self.sigma, jit=self.jit)
            self.k_steps_eval_fn = partial(k_steps_eval_osqp, P=self.P,
                                           A=self.A, rho=self.rho, sigma=self.sigma, jit=self.jit, 
                                           custom_loss=custom_loss)
        else:
            self.k_steps_train_fn = self.create_k_steps_train_fn_dynamic()
            self.k_steps_eval_fn = self.create_k_steps_eval_fn_dynamic()
            # self.k_steps_eval_fn = partial(k_steps_eval_osqp, rho=rho, sigma=sigma, jit=self.jit)

            self.factors_train = input_dict['factors_train']
            self.factors_test = input_dict['factors_test']

            

        # self.k_steps_train_fn = partial(k_steps_train_osqp, factor=factor, A=self.A, rho=rho, 
        #                                 sigma=sigma, jit=self.jit)
        # self.k_steps_eval_fn = partial(k_steps_eval_osqp, factor=factor, P=self.P, A=self.A, 
        #                                rho=rho, sigma=sigma, jit=self.jit)
        self.out_axes_length = 6

    def create_k_steps_train_fn_dynamic(self):
        """
        creates the self.k_steps_train_fn function for the dynamic case
        acts as a wrapper around the k_steps_train_osqp functino from algo_steps.py

        we want to maintain the argument inputs as (k, z0, q_bar, factor, supervised, z_star)
        """
        m, n = self.m, self.n

        def k_steps_train_osqp_dynamic(k, z0, q, factor, supervised, z_star):
            nc2 = int(n * (n + 1) / 2)
            q_bar = q[:2 * m + n]
            unvec_symm(q[2 * m + n: 2 * m + n + nc2], n)
            A = jnp.reshape(q[2 * m + n + nc2:], (m, n))
            return k_steps_train_osqp(k=k, z0=z0, q=q_bar,
                                      factor=factor, A=A, rho=self.rho, sigma=self.sigma,
                                      supervised=supervised, z_star=z_star, jit=self.jit)
        return k_steps_train_osqp_dynamic

    def create_k_steps_eval_fn_dynamic(self):
        """
        creates the self.k_steps_train_fn function for the dynamic case
        acts as a wrapper around the k_steps_train_osqp functino from algo_steps.py

        we want to maintain the argument inputs as (k, z0, q_bar, factor, supervised, z_star)
        """
        m, n = self.m, self.n

        def k_steps_eval_osqp_dynamic(k, z0, q, factor, supervised, z_star):
            nc2 = int(n * (n + 1) / 2)
            q_bar = q[:2 * m + n]
            P = unvec_symm(q[2 * m + n: 2 * m + n + nc2], n)
            A = jnp.reshape(q[2 * m + n + nc2:], (m, n))
            return k_steps_eval_osqp(k=k, z0=z0, q=q_bar,
                                     factor=factor, P=P, A=A, rho=self.rho, sigma=self.sigma,
                                     supervised=supervised, z_star=z_star, jit=self.jit)
        return k_steps_eval_osqp_dynamic
