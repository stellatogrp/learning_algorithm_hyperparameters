from functools import partial

import jax.numpy as jnp
from jax import random

import numpy as np

from lah.algo_steps import k_steps_eval_lm_gd, k_steps_train_lm_gd, k_steps_eval_nesterov_gd
from lah.l2ws_model import L2WSmodel
from lah.utils.nn_utils import calculate_pinsker_penalty, compute_single_param_KL
from lah.lah_gd_model import sigmoid


class LMGDmodel(L2WSmodel):
    def __init__(self, **kwargs):
        super(LMGDmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        self.factor_static = None
        self.algo = 'lm_gd'
        self.factors_required = False
        self.q_mat_train, self.q_mat_test = input_dict['c_mat_train'], input_dict['c_mat_test']

        P = input_dict['P']
        self.P = P

        # self.D, self.W = D, W
        self.n = P.shape[0]
        self.output_size = self.n


        evals, evecs = jnp.linalg.eigh(P)

        self.str_cvx_param = jnp.min(evals)
        self.smooth_param = jnp.max(evals)

        # cond_num = self.smooth_param / self.str_cvx_param

        self.k_steps_train_fn = partial(k_steps_train_lm_gd, P=P,
                                        jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_lm_gd, P=P,
                                       jit=self.jit)

        self.out_axes_length = 5

        self.lah_train_inputs = self.q_mat_train

        e2e_loss_fn = self.create_end2end_loss_fn

        self.lah = False
        self.lm = True



        # end-to-end loss fn for silver evaluation
        # self.loss_fn_eval_silver = e2e_loss_fn(bypass_nn=False, diff_required=False, 
        #                                        special_algo='silver')

        # end-to-end added fixed warm start eval - bypasses neural network
        # self.loss_fn_fixed_ws = e2e_loss_fn(bypass_nn=True, diff_required=False)



    # def init_params(self):
        # # init step-varying params
        # step_varying_params = jnp.log(2 / (self.smooth_param + self.str_cvx_param)) * jnp.ones((self.step_varying_num, 1))

        # # init steady_state_params
        # steady_state_params = sigmoid_inv(self.smooth_param / (self.smooth_param + self.str_cvx_param)) * jnp.ones((1, 1))

        # self.params = [jnp.vstack([step_varying_params, steady_state_params])]
        # sigmoid_inv(beta)

        # self.params = [jnp.log(2 / (self.smooth_param + self.str_cvx_param)) * jnp.ones((self.n, 1))]


    def create_end2end_loss_fn(self, bypass_nn, diff_required, special_algo='gd'):
        supervised = True  # self.supervised and diff_required
        loss_method = self.loss_method

        def predict(params, input, q, iters, z_star, key, factor):
            z0 = 0 * z_star

            gd_step = sigmoid(jnp.exp(self.predict_warm_start(params, input, key, bypass_nn))) * 2 / self.smooth_param

            # import pdb
            # pdb.set_trace()

            # if diff_required:
            #     n_iters = key #self.train_unrolls if key else 1
            #     stochastic_params = jnp.exp(params[0])
            # else:
            #     n_iters = key
            #     stochastic_params = jnp.exp(params[0])

            if self.train_fn is not None:
                train_fn = self.train_fn
            else:
                train_fn = self.k_steps_train_fn
            if self.eval_fn is not None:
                eval_fn = self.eval_fn
            else:
                eval_fn = self.k_steps_eval_fn

            # stochastic_params = params[0][:n_iters, 0]
            if diff_required:
                z_final, iter_losses = train_fn(k=iters,
                                                z0=z0,
                                                q=q,
                                                params=gd_step,
                                                supervised=supervised,
                                                z_star=z_star)
            else:
                eval_out = eval_fn(k=iters,
                                    z0=z0,
                                    q=q,
                                    params=gd_step,
                                    supervised=supervised,
                                    z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                angles = None

            loss = self.final_loss(loss_method, z_final,
                                   iter_losses, supervised, z0, z_star)

            if diff_required:
                return loss
            else:
                return_out = (loss, iter_losses, z_all_plus_1,
                              angles) + eval_out[3:]
                return return_out
        loss_fn = self.predict_2_loss(predict, diff_required)
        return loss_fn

    def calculate_total_penalty(self, N_train, params, c, b, delta):
        return 0
        # priors are already rounded
        rounded_priors = params[2]

        # second: calculate the penalties
        num_groups = len(rounded_priors)
        pi_pen = jnp.log(jnp.pi ** 2 * num_groups * N_train / (6 * delta))
        log_pen = 0
        for i in range(num_groups):
            curr_lambd = jnp.clip(jnp.exp(rounded_priors[i]), a_max=c)
            log_pen += 2 * jnp.log(b * jnp.log((c+1e-6) / curr_lambd))

        # calculate the KL penalty
        penalty_loss = self.compute_all_params_KL(params[0], params[1],
                                                  rounded_priors) + pi_pen + log_pen
        return penalty_loss / N_train

    def compute_all_params_KL(self, mean_params, sigma_params, lambd):
        return 0
        # step size
        total_pen = compute_single_param_KL(
            mean_params, jnp.exp(sigma_params), jnp.exp(lambd[0]))

        # # threshold
        # total_pen += compute_single_param_KL(mean_params, jnp.exp(sigma_params), jnp.exp(lambd[1]))
        return total_pen

    def compute_weight_norm_squared(self, nn_params):
        return jnp.linalg.norm(nn_params) ** 2, nn_params.size

    def calculate_avg_posterior_var(self, params):
        return 0, 0
        sigma_params = params[1]
        flattened_params = jnp.concatenate([jnp.ravel(weight_matrix) for weight_matrix, _ in sigma_params] +
                                           [jnp.ravel(bias_vector) for _, bias_vector in sigma_params])
        variances = jnp.exp(flattened_params)
        avg_posterior_var = variances.mean()
        stddev_posterior_var = variances.std()
        return avg_posterior_var, stddev_posterior_var

