from functools import partial

import jax.numpy as jnp
from jax import random

import numpy as np

from lah.algo_steps_ista import k_steps_eval_lah_ista, k_steps_train_lah_ista, k_steps_eval_fista
from lah.l2ws_model import L2WSmodel
from lah.utils.nn_utils import calculate_pinsker_penalty, compute_single_param_KL

from jax import vmap, jit


class LAHISTAmodel(L2WSmodel):
    def __init__(self, **kwargs):
        super(LAHISTAmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        self.factor_static = None
        self.algo = 'lah_ista'
        self.factors_required = False
        self.q_mat_train, self.q_mat_test = input_dict['c_mat_train'], input_dict['c_mat_test']
        # self.q_mat_train, self.q_mat_test = input_dict['b_mat_train'], input_dict['b_mat_test']
        # D, W = input_dict['D'], input_dict['W']
        A = input_dict['A']
        lambd = input_dict['lambd']
        self.A = A
        self.lambd = lambd

        # self.D, self.W = D, W
        self.m, self.n = A.shape
        self.output_size = self.n

        # evals, evecs = jnp.linalg.eigh(D.T @ D)
        # lambd = 0.1
        # self.ista_step = lambd / evals.max()
        # p = jnp.diag(P)
        # cond_num = jnp.max(p) / jnp.min(p)
        evals, evecs = jnp.linalg.eigh(A.T @ A)

        self.str_cvx_param = jnp.min(evals)
        self.smooth_param = jnp.max(evals)

        cond_num = self.smooth_param / self.str_cvx_param

        safeguard_step = 1 / self.smooth_param

        self.k_steps_train_fn = partial(k_steps_train_lah_ista, lambd=lambd, A=A,
                                        jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_lah_ista, lambd=lambd, A=A, 
                                       safeguard_step=jnp.array([safeguard_step]),
                                       jit=self.jit)
        self.nesterov_eval_fn = partial(k_steps_eval_fista, lambd=lambd, A=A,
                                       jit=self.jit)
        # self.conj_grad_eval_fn = partial(k_steps_eval_conj_grad, P=P,
        #                                jit=self.jit)
        self.out_axes_length = 5

        N = self.q_mat_train.shape[0]
        self.lah_train_inputs = jnp.zeros((N, self.n))


        e2e_loss_fn = self.create_end2end_loss_fn



        # end-to-end loss fn for silver evaluation
        # self.loss_fn_eval_silver = e2e_loss_fn(bypass_nn=False, diff_required=False, 
        #                                        special_algo='silver')
        
        # end-to-end loss fn for silver evaluation
        # self.loss_fn_eval_conj_grad = e2e_loss_fn(bypass_nn=False, diff_required=False, 
        #                                        special_algo='conj_grad')

        # end-to-end added fixed warm start eval - bypasses neural network
        # self.loss_fn_fixed_ws = e2e_loss_fn(bypass_nn=True, diff_required=False)

    def f(self, z, c):
        return .5 * jnp.linalg.norm(self.A @ z - c) ** 2 + self.lambd * jnp.linalg.norm(z, ord=1)

    # def compute_avg_opt(self):
    #     batch_f = vmap(self.f, in_axes=(0, 0), out_axes=(0))
    #     opt_vals = batch_f(self.z_stars_train, self.theta)

    def transform_params(self, params, n_iters):
        # n_iters = params[0].size
        transformed_params = jnp.zeros((n_iters, 1))
        transformed_params = transformed_params.at[:n_iters - 1, 0].set(jnp.exp(params[0][:n_iters - 1, 0]))
        transformed_params = transformed_params.at[n_iters - 1, 0].set(2 / self.smooth_param * sigmoid(params[0][n_iters - 1, 0]))
        # transformed_params = jnp.clip(transformed_params, a_max=1.0)
        return transformed_params

    # def perturb_params(self):
    #     # init step-varying params
    #     noise = jnp.array(np.clip(np.random.normal(size=(self.step_varying_num, 1)), a_min=1e-5, a_max=1e0)) * 0.00001
    #     step_varying_params = jnp.log(noise + 2 / (self.smooth_param + self.str_cvx_param)) * jnp.ones((self.step_varying_num, 1))

    #     # init steady_state_params
    #     steady_state_params = sigmoid_inv(self.smooth_param / (self.smooth_param + self.str_cvx_param)) * jnp.ones((1, 1))

    #     self.params = [jnp.vstack([step_varying_params, steady_state_params])]


    def set_params_for_nesterov(self):
        self.init_params()
        # self.params = [jnp.log(1 / self.smooth_param * jnp.ones((self.step_varying_num + 1, 1)))]


    # def set_params_for_silver(self):
    #     silver_steps = 128
    #     kappa = self.smooth_param / self.str_cvx_param
    #     silver_step_sizes = compute_silver_steps(kappa, silver_steps) / self.smooth_param
    #     params = jnp.ones((silver_steps + 1, 1))
    #     params = params.at[:silver_steps, 0].set(jnp.array(silver_step_sizes))
    #     params = params.at[silver_steps, 0].set(2 / (self.smooth_param + self.str_cvx_param))

    #     self.params = [params]
        # step_varying_params = jnp.log(params[:self.step_varying_num, :1])
        # steady_state_params = sigmoid_inv(params[self.step_varying_num:, :1] * self.smooth_param / 2)
        # self.params = [jnp.vstack([step_varying_params, steady_state_params])]


    def init_params(self):
        # init step-varying params
        step_varying_params = jnp.log(1 / self.smooth_param) * jnp.ones((self.step_varying_num, 1))

        # init steady_state_params
        steady_state_params = 0 * jnp.ones((1, 1)) #sigmoid_inv(1 / (self.smooth_param)) * jnp.ones((1, 1))

        self.params = [jnp.vstack([step_varying_params, steady_state_params])]
        # sigmoid_inv(beta)



    def create_end2end_loss_fn(self, bypass_nn, diff_required, special_algo='gd'):
        supervised = True  # self.supervised and diff_required
        loss_method = self.loss_method

        @partial(jit, static_argnames=['iters', 'key'])
        def predict(params, input, q, iters, z_star, key, factor):
            z0 = input
            if diff_required:
                n_iters = key #self.train_unrolls if key else 1

                if n_iters == 1:
                    # for steady_state training
                    stochastic_params = 2 / self.smooth_param * sigmoid(params[0][:n_iters, 0])
                else:
                    # for step-varying training
                    stochastic_params = jnp.exp(params[0][:n_iters, 0])
            else:
                if special_algo == 'silver' or special_algo == 'conj_grad':
                    stochastic_params = params[0]
                else:
                    n_iters = key #min(iters, 51)
                    # stochastic_params = jnp.zeros((n_iters, 1))
                    # stochastic_params = stochastic_params.at[:n_iters - 1, 0].set(jnp.exp(params[0][:n_iters - 1, 0]))
                    # stochastic_params = stochastic_params.at[n_iters - 1, 0].set(2 / self.smooth_param * sigmoid(params[0][n_iters - 1, 0]))
                    stochastic_params = self.transform_params(params, n_iters)

            # stochastic_params = jnp.clip(stochastic_params, a_max=0.3)
            if self.train_fn is not None:
                train_fn = self.train_fn
            else:
                train_fn = self.k_steps_train_fn
            if self.eval_fn is not None:
                eval_fn = self.eval_fn
            else:
                eval_fn = self.k_steps_eval_fn



            # stochastic_params = params[0][:n_iters, 0]
            if special_algo == 'nesterov':
                eval_out = self.nesterov_eval_fn(k=iters,
                                   z0=z0,
                                   q=q,
                                   params=stochastic_params,
                                   supervised=supervised,
                                   z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]
                angles = None
            elif bypass_nn:
                # use nesterov's acceleration
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

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def sigmoid_inv(beta):
    return jnp.log(beta / (1 - beta))