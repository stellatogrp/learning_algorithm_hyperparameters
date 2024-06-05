from functools import partial

import jax.numpy as jnp
from jax import random

import numpy as np

from lasco.algo_steps import k_steps_eval_lasco_gd, k_steps_train_lasco_gd, k_steps_eval_nesterov_gd
from lasco.l2ws_model import L2WSmodel
from lasco.utils.nn_utils import calculate_pinsker_penalty, compute_single_param_KL


class LASCOGDmodel(L2WSmodel):
    def __init__(self, **kwargs):
        super(LASCOGDmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        self.factor_static = None
        self.algo = 'lasco_gd'
        self.factors_required = False
        self.q_mat_train, self.q_mat_test = input_dict['c_mat_train'], input_dict['c_mat_test']
        # self.q_mat_train, self.q_mat_test = input_dict['b_mat_train'], input_dict['b_mat_test']
        # D, W = input_dict['D'], input_dict['W']
        P = input_dict['P']
        self.P = P

        # self.D, self.W = D, W
        self.n = P.shape[0]
        self.output_size = self.n

        # evals, evecs = jnp.linalg.eigh(D.T @ D)
        # lambd = 0.1
        # self.ista_step = lambd / evals.max()
        # p = jnp.diag(P)
        # cond_num = jnp.max(p) / jnp.min(p)
        evals, evecs = jnp.linalg.eigh(P)

        self.str_cvx_param = jnp.min(evals)
        self.smooth_param = jnp.max(evals)

        cond_num = self.smooth_param / self.str_cvx_param

        self.k_steps_train_fn = partial(k_steps_train_lasco_gd, P=P,
                                        jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_lasco_gd, P=P,
                                       jit=self.jit)
        self.nesterov_eval_fn = partial(k_steps_eval_nesterov_gd, P=P, cond_num=cond_num,
                                       jit=self.jit)
        self.out_axes_length = 5

        self.lasco_train_inputs = self.q_mat_train

    def perturb_params(self):
        noise = np.clip(np.random.normal(size=(self.eval_unrolls, 1)), a_min=1e-5, a_max=1e0)
        self.mean_params = 2 / (self.smooth_param + self.str_cvx_param) * jnp.ones((self.eval_unrolls, 1))
        params = self.mean_params + .001 * jnp.array(noise)

        params = params.at[51:, 0].set(0)
        self.params = [params]
        # import pdb
        # pdb.set_trace()

    def set_params_for_nesterov(self):
        self.params = [1 / self.smooth_param * jnp.ones((self.eval_unrolls, 1))]


    def set_params_for_silver(self):
        kappa = self.smooth_param / self.str_cvx_param
        silver_step_sizes = compute_silver_steps(kappa, 64) / self.smooth_param
        params = jnp.ones((64, 1))
        params = params.at[:, 0].set(jnp.array(silver_step_sizes))
        self.params = [params]
        # import pdb
        # pdb.set_trace()



    def init_params(self):
        # w_key = random.PRNGKey(0)
        # self.mean_params = random.normal(w_key, (self.train_unrolls, 2))
        # self.mean_params = self.mean_params[:, 0]
        # self.mean_params =
        # p = jnp.diag(self.P)
        noise = np.clip(np.random.normal(size=(self.eval_unrolls, 1)) / 10, a_min=0.0001, a_max=1e10)
        # self.mean_params = 2 / (p.max() + p.min()) * jnp.ones((self.eval_unrolls, 1))  #+ 1 * jnp.array(noise)
        # self.mean_params = 1 / p.max() * jnp.ones((self.eval_unrolls, 1))
        # self.mean_params = 1 / self.smooth_param * jnp.ones((self.eval_unrolls, 1)) + .001 * jnp.array(noise)
        self.mean_params = 2 / (self.smooth_param + self.str_cvx_param) * jnp.ones((self.eval_unrolls, 1)) #+ .001 * jnp.array(noise)
        print('self.mean_params', self.mean_params)
        # self.mean_params = self.mean_params.at[2].set(0.1)

        # self.sigma_params = -jnp.ones(self.train_unrolls) * 10

        # initialize the prior
        # self.prior_param = jnp.log(self.init_var) * jnp.ones(2)

        # , self.prior_param]
        self.params = [self.mean_params]


    def create_end2end_loss_fn(self, bypass_nn, diff_required):
        supervised = True  # self.supervised and diff_required
        loss_method = self.loss_method

        def predict(params, input, q, iters, z_star, key, factor):
            # if diff_required:
            #     z0 = input
            # else:
            #     z0 = jnp.zeros(z_star.size)
            z0 = input
            if diff_required:
                n_iters = key #self.train_unrolls if key else 1
            else:
                n_iters = min(iters, 51)

            # if bypass_nn:
            #     eval_fn = self.nesterov_eval_fn

            if self.train_fn is not None:
                train_fn = self.train_fn
            else:
                train_fn = self.k_steps_train_fn
            if self.eval_fn is not None:
                eval_fn = self.eval_fn
            else:
                eval_fn = self.k_steps_eval_fn

            stochastic_params = params[0][:n_iters, 0]

            if bypass_nn:
                eval_out = self.nesterov_eval_fn(k=iters,
                                   z0=z0,
                                   q=q,
                                   params=stochastic_params,
                                   supervised=supervised,
                                   z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                angles = None
            else:
                if diff_required:
                    z_final, iter_losses = train_fn(k=iters,
                                                    z0=z0,
                                                    q=q,
                                                    params=stochastic_params,
                                                    supervised=supervised,
                                                    z_star=z_star)
                else:
                    eval_out = eval_fn(k=iters,
                                       z0=z0,
                                       q=q,
                                       params=stochastic_params,
                                       supervised=supervised,
                                       z_star=z_star)
                    z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                    angles = None

            loss = self.final_loss(loss_method, z_final,
                                   iter_losses, supervised, z0, z_star)

            # penalty_loss = calculate_pinsker_penalty(self.N_train, params, self.b, self.c, self.delta)
            # loss = loss + self.penalty_coeff * penalty_loss

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

def generate_yz_sequences(kappa, t):
    # intended to be t = log_2(K)
    y_vals = {1: 1 / kappa}
    z_vals = {1: 1 / kappa}
    print(y_vals, z_vals)

    # first generate z sequences
    for i in range(1, t+1):
        K = 2 ** i
        z_ihalf = z_vals[int(K / 2)]
        xi = 1 - z_ihalf
        z_i = z_ihalf * (xi + np.sqrt(1 + xi ** 2))
        z_vals[K] = z_i

    for i in range(1, t+1):
        K = 2 ** i
        # z_ihalf = z_vals[int(K / 2)]
        # xi = 1 - z_ihalf
        # yi = z_ihalf / (xi + np.sqrt(1 + xi ** 2))
        # y_vals[K] = yi
        zK = z_vals[K]
        zKhalf = z_vals[int(K // 2)]
        yK = zK - 2 * (zKhalf - zKhalf ** 2)
        y_vals[K] = yK

    # print(y_vals, z_vals)

    # print(z_vals[1], z_vals[2])
    # print((1 / kappa ** 2) / z_vals[2])
    return y_vals, z_vals

def compute_silver_steps(kappa, K):
    # assume K is a power of 2
    idx_vals = compute_silver_idx(kappa, K)
    y_vals, z_vals = generate_yz_sequences(kappa, int(np.log2(K)))

    def psi(t):
        return (1 + kappa * t) / (1 + t)

    # print(y_vals, z_vals)
    silver_steps = []
    for i in range(idx_vals.shape[0] - 1):
        idx = idx_vals[i]
        silver_steps.append(psi(y_vals[idx]))
    silver_steps.append(psi(z_vals[idx_vals[-1]]))
    print(silver_steps)

    return np.array(silver_steps)

def compute_silver_idx(kappa, K):
    two_adics = compute_shifted_2adics(K)
    # print(two_adics)
    idx_vals = np.power(2, two_adics)

    # if np.ceil(np.log2(K)) == np.floor(np.log2(K)):
    last_pow2 = int(np.floor(np.log2(K)))
    # print(last_pow2)
    idx_vals[(2 ** last_pow2) - 1] /= 2
    print('a_idx:', idx_vals)
    return idx_vals

def compute_shifted_2adics(K):
    return np.array([(k & -k).bit_length() for k in range(1, K+1)])