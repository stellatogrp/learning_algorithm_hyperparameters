from functools import partial

import jax.numpy as jnp
from jax import random

from lasco.algo_steps import k_steps_eval_lasco_gd, k_steps_train_lasco_gd
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

        self.k_steps_train_fn = partial(k_steps_train_lasco_gd, P=P,
                                        jit=self.jit)
        self.k_steps_eval_fn = partial(k_steps_eval_lasco_gd, P=P,
                                       jit=self.jit)
        self.out_axes_length = 5


    def init_params(self):
        # w_key = random.PRNGKey(0)
        # self.mean_params = random.normal(w_key, (self.train_unrolls, 2))
        # self.mean_params = self.mean_params[:, 0]
        # self.mean_params = 
        p = jnp.diag(self.P)
        self.mean_params = (p.max() + p.min()) / 2 * jnp.ones(self.train_unrolls)
        # self.mean_params = self.mean_params.at[2].set(0.1)

        self.sigma_params = -jnp.ones(self.train_unrolls) * 10

        # initialize the prior
        self.prior_param = jnp.log(self.init_var) * jnp.ones(2)

        self.params = [self.mean_params, self.sigma_params] #, self.prior_param]


    def create_end2end_loss_fn(self, bypass_nn, diff_required):
        supervised = True #self.supervised and diff_required
        loss_method = self.loss_method

        def predict(params, input, q, iters, z_star, key, factor):
            z0 = jnp.zeros(z_star.size)

            if self.train_fn is not None:
                train_fn = self.train_fn
            else:
                train_fn = self.k_steps_train_fn
            if self.eval_fn is not None:
                eval_fn = self.eval_fn
            else:
                eval_fn = self.k_steps_eval_fn

            # w_key = random.split(key)
            w_key = random.PRNGKey(key)
            perturb = random.normal(w_key, (self.train_unrolls, 2))
            if self.deterministic:
                stochastic_params = params[0]
            else:
                stochastic_params = params[0] + jnp.sqrt(jnp.exp(params[1])) * perturb
            print('stochastic_params', stochastic_params)
            print('self.deterministic', self.deterministic)
            print('iters', iters)
        

            if bypass_nn:
                eval_out = eval_fn(k=iters,
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

            loss = self.final_loss(loss_method, z_final, iter_losses, supervised, z0, z_star)

            # penalty_loss = calculate_pinsker_penalty(self.N_train, params, self.b, self.c, self.delta)
            # loss = loss + self.penalty_coeff * penalty_loss

            if diff_required:
                return loss
            else:
                return_out = (loss, iter_losses, z_all_plus_1, angles) + eval_out[3:]
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
        return penalty_loss /  N_train


    def compute_all_params_KL(self, mean_params, sigma_params, lambd):
        return 0
        # step size
        total_pen = compute_single_param_KL(mean_params, jnp.exp(sigma_params), jnp.exp(lambd[0]))

        # # threshold
        # total_pen += compute_single_param_KL(mean_params, jnp.exp(sigma_params), jnp.exp(lambd[1]))
        return total_pen


    def compute_weight_norm_squared(self, nn_params):
        return jnp.linalg.norm(nn_params) ** 2, nn_params.size

    
    def calculate_avg_posterior_var(self, params):
        return 0,0
        sigma_params = params[1]
        flattened_params = jnp.concatenate([jnp.ravel(weight_matrix) for weight_matrix, _ in sigma_params] + 
                                        [jnp.ravel(bias_vector) for _, bias_vector in sigma_params])
        variances = jnp.exp(flattened_params)
        avg_posterior_var = variances.mean()
        stddev_posterior_var = variances.std()
        return avg_posterior_var, stddev_posterior_var
    
