from functools import partial

from lah.algo_steps_logistic import k_steps_eval_logisticgd, k_steps_train_logisticgd
from lah.l2ws_model import L2WSmodel
import jax.numpy as jnp
import numpy as np


class LOGISTICGDmodel(L2WSmodel):
    def __init__(self, **kwargs):
        super(LOGISTICGDmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        self.factor_static = None
        self.algo = 'logisticgd'
        self.factors_required = False

        num_points = input_dict['num_points']
        num_weights = 785
        
        self.q_mat_train, self.q_mat_test = input_dict['q_mat_train'], input_dict['q_mat_test']
        # gd_step = input_dict['gd_step']
        self.n = num_weights
        self.output_size = self.n


        # get the smoothness parameter
        first_prob_data = self.q_mat_train[0, :]
        X0_flat, y0 = first_prob_data[:-num_points], first_prob_data[-num_points:]
        X0 = jnp.reshape(X0_flat, (num_points, num_weights - 1))

        covariance_matrix = np.dot(X0.T, X0) / num_points
    
        # Compute the maximum eigenvalue of the covariance matrix
        evals, evecs = jnp.linalg.eigh(covariance_matrix)
        # max_eigenvalue = jnp.max(evals)

        self.num_const_steps = input_dict.get('num_const_steps', 1)

        # evals, evecs = jnp.linalg.eigh(P)

        # self.str_cvx_param = jnp.min(evals)
        self.smooth_param = jnp.max(evals) / 4

        gd_step = 1 / self.smooth_param

        self.k_steps_eval_fn = partial(k_steps_eval_logisticgd, num_points=num_points, gd_step=gd_step, jit=self.jit)
        self.k_steps_train_fn = partial(k_steps_train_logisticgd, num_points=num_points, gd_step=gd_step, jit=self.jit)
        self.out_axes_length = 5
        self.lasco = False
