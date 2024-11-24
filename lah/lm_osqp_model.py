from functools import partial

import cvxpy as cp
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scs
from scipy.sparse import csc_matrix
import jax

from lah.algo_steps_osqp import (
    k_steps_eval_lm_osqp,
    k_steps_train_lm_osqp
)
from lah.l2ws_model import L2WSmodel
from lah.utils.nn_utils import (
    predict_y
)


class LMOSQPmodel(L2WSmodel):
    def __init__(self, **kwargs):
        super(LMOSQPmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        """
        the input_dict is required to contain these keys
        otherwise there is an error
        """
        self.factors_required = True
        self.factor_static_bool = input_dict.get('factor_static_bool', True)
        self.algo = 'lm_osqp'
        self.factors_required = True
        # self.hsde = input_dict.get('hsde', True)
        self.m, self.n = input_dict['m'], input_dict['n']
        # self.cones = input_dict['cones']
        self.static_flag = input_dict.get('static_flag', True)
        self.q_mat_train, self.q_mat_test = input_dict['q_mat_train'], input_dict['q_mat_test']

        # M = input_dict['static_M']
        self.P = input_dict['P'] #M[:self.n, :self.n]
        self.A = input_dict['A'] #-M[self.n:, :self.n]
        # self.M = M


        lightweight = input_dict.get('lightweight', False)

        self.output_size = self.n + self.m #+ 1
        self.out_axes_length = 6

        # custom_loss
        custom_loss = input_dict.get('custom_loss')

        self.k_steps_train_fn = partial(k_steps_train_lm_osqp,
                                        jit=self.jit,
                                        P=self.P,
                                        A=self.A)
        # self.k_steps_train_fn2 = partial(k_steps_train_lm_scs, proj=self.proj,
        #                                 jit=self.jit,
        #                                 P=self.P,
        #                                 A=self.A,
        #                                 hsde=False)
        self.k_steps_eval_fn = partial(k_steps_eval_lm_osqp,
                                       P=self.P, A=self.A,
                                    #    zero_cone_size=self.zero_cone_size,
                                       jit=self.jit) #,
                                    #    custom_loss=custom_loss,
                                    #    lightweight=lightweight)
        self.lm = True
        self.lah = False
        
        
    # def init_params(self):
    #     self.mean_params = 0*jnp.ones((self.eval_unrolls, 5))
    #     # self.mean_params = self.mean_params.at[10:20, :].set(1*jnp.ones((10, 5)))
    #     self.params = [self.mean_params]


    def setup_optimal_solutions(self, 
                                z_stars_train, 
                                z_stars_test, 
                                x_stars_train=None, 
                                x_stars_test=None, 
                                y_stars_train=None, 
                                y_stars_test=None):
        if x_stars_train is not None:
            self.z_stars_train = jnp.array(z_stars_train)
            self.z_stars_test = jnp.array(z_stars_test)
            self.x_stars_train = jnp.array(x_stars_train)
            self.x_stars_test = jnp.array(x_stars_test)
            self.y_stars_train = jnp.array(y_stars_train)
            self.y_stars_test = jnp.array(y_stars_test)
            self.u_stars_train = jnp.hstack([self.x_stars_train, self.y_stars_train])
            self.u_stars_test = jnp.hstack([self.x_stars_test, self.y_stars_test])
        if z_stars_train is not None:
            self.z_stars_train = jnp.array(z_stars_train)
            self.z_stars_test = jnp.array(z_stars_test)
        else:
            self.z_stars_train, self.z_stars_test = None, None
        self.lah_train_inputs = jnp.hstack([0 * self.z_stars_train, jnp.ones((self.z_stars_train.shape[0], 1))])


    def predict_scale_vec(self, params, input, key, bypass_nn):
        """
        gets the warm-start
        bypass_nn means we ignore the neural network and set z0=input
        """
        scale_vec = predict_y(params[0], input)

        # scale_vec = jnp.zeros(self.m + self.n + 1)
        # num_scalar_cone = self.n + self.zero_cone_size + self.ineq_cone_size
        # scale_vec = scale_vec.at[:num_scalar_cone].set(nn_output[:num_scalar_cone])

        return scale_vec

    def create_end2end_loss_fn(self, bypass_nn, diff_required):
        supervised = self.supervised and diff_required
        loss_method = self.loss_method

        def predict(params, input, q, iters, z_star, key, factor):
            # q = lin_sys_solve(factor, q)
            z0 = jnp.zeros(self.m + self.n) #self.predict_warm_start(params, input, key, bypass_nn)
            # z0 = z0.at[-1].set(1)

            if self.train_fn is not None:
                train_fn = self.train_fn
            else:
                train_fn = self.k_steps_train_fn
            if self.eval_fn is not None:
                eval_fn = self.eval_fn
            else:
                eval_fn = self.k_steps_eval_fn

            # predict the metric
            scale_vec = jnp.exp(self.predict_scale_vec(params, input, key, bypass_nn))
            sigma_vec = scale_vec[:self.n]
            rho_vec = scale_vec[self.n:]
            # scale_vec, tau_factor = scale_vec_and_tau[:-1], scale_vec_and_tau[-1]
            # scale_vec = scale_vec.at[self.n + self.zero_cone_size:].set(scale_vec[self.n + self.zero_cone_size])
            M = self.P + jnp.diag(sigma_vec) + self.A.T @ jnp.diag(rho_vec) @ self.A
            factor = jsp.linalg.lu_factor(M)

            # print('scale_vec', scale_vec)
            # print('tau_factor', tau_factor)
            # import pdb
            # pdb.set_trace()
            # do all of the factorizations
            # factors1 = jnp.zeros((n_iters, self.m + self.n, self.m + self.n))
            # factors2 = jnp.zeros((n_iters, self.m + self.n), dtype=jnp.int32)
            # scaled_vecs = jnp.zeros((n_iters, self.m + self.n))
            # rho_xs, rho_ys, rho_ys_zero = params[0][:, 0], params[0][:, 1], params[0][:, 4]
            # for i in range(n_iters):
            #     rho_x, rho_y = jnp.exp(rho_xs[i]), jnp.exp(rho_ys[i])
            #     rho_y_zero = jnp.exp(rho_ys_zero[i])
                
            #     # needs to be different
            #     # factor, scale_vec = get_scaled_vec_and_factor(self.M, rho_x, rho_y, 
            #     #                                               rho_y_zero,
            #     #                                               self.m, self.n, 
            #     #                                               zero_cone_size=self.zero_cone_size, 
            #     #                                               hsde=True)
            #     factors1 = factors1.at[i, :, :].set(factor[0])
            #     factors2 = factors2.at[i, :].set(factor[1])
            #     scaled_vecs = scaled_vecs.at[i, :].set(scale_vec)
            # all_factors = factors1, factors2
            osqp_params = (factor, scale_vec)
            if diff_required:
                z_final, iter_losses = train_fn(k=iters,
                                                z0=z0,
                                                q=q,
                                                params=osqp_params,
                                                supervised=supervised,
                                                z_star=z_star)
            else:
                eval_out = eval_fn(k=iters,
                                    z0=z0,
                                    q=q,
                                    params=osqp_params,
                                    supervised=supervised,
                                    z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                
                angles = None

            loss = self.final_loss(loss_method, z_final, iter_losses, supervised, z0, z_star)

            if diff_required:
                return loss
            else:
                return_out = (loss, iter_losses, z_all_plus_1, angles) + eval_out[3:]
                return return_out
        loss_fn = self.predict_2_loss(predict, diff_required)
        return loss_fn

    
    def get_xys_from_z(self, z_init, m, n):
        """
        z = (x, y + s, 1)
        we always set the last entry of z to be 1
        we allow s to be zero (we just need z[n:m + n] = y + s)
        """
        # m, n = self.l2ws_model.m, self.l2ws_model.n
        x = z_init[:n]
        y = z_init[n:n + m]
        s = jnp.zeros(m)
        return x, y, s
