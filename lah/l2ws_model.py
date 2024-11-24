import logging
import time
from functools import partial

import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, random, vmap
# from jax.config import config
from jaxopt import OptaxSolver

from lah.low_step_solvers import two_step_data_quad_gd_solver, one_step_gd_solver, one_step_prox_gd_solver, three_step_data_quad_gd_solver, two_step_stochastic_quad_gd_solver, three_step_stochastic_quad_gd_solver, stochastic_get_z_bar, one_step_stochastic_quad_gd_solver
from lah.algo_steps import create_eval_fn, create_train_fn, lin_sys_solve, create_kl_inv_layer, kl_inv_fn
from lah.utils.nn_utils import (
    calculate_pinsker_penalty,
    # calculate_total_penalty,
    get_perturbed_weights,
    init_network_params,
    init_variance_network_params,
    predict_y,
    compute_single_param_KL
)
from jaxopt import Bisection


import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_disable_jit', True)
# jax.config.update("jax_debug_nans", True)


class L2WSmodel(object):
    def __init__(self, 
                 train_unrolls=5,
                 step_varying_num=50,
                 train_inputs=None,
                 test_inputs=None,
                 regression=False,
                 nn_cfg={},
                 pac_bayes_cfg={},
                 plateau_decay={},
                 jit=True,
                 eval_unrolls=500,
                 z_stars_train=None,
                 z_stars_test=None,
                 x_stars_train=None,
                 x_stars_test=None,
                 y_stars_train=None,
                 y_stars_test=None,
                 loss_method='fixed_k',
                 algo_dict={}):
        self.train_case = 'gradient'
        dict = algo_dict
        self.key = 0
        self.sigma = 0.01
        self.b = pac_bayes_cfg.get('b', 100)
        self.c = pac_bayes_cfg.get('c', 2.0)
        self.delta = pac_bayes_cfg.get('delta', 0.00001)
        self.delta2 = pac_bayes_cfg.get('delta', 0.00001)
        # self.target_pen = pac_bayes_cfg['target_pen']
        self.init_var = pac_bayes_cfg.get('init_var', 1e-1) # initializes all of s and the prior
        self.penalty_coeff = pac_bayes_cfg.get('penalty_coeff', 1.0)
        self.deterministic = pac_bayes_cfg.get('deterministic', False)
        self.prior = 0

        # essential pieces for the model
        self.initialize_essentials(jit, eval_unrolls, train_unrolls, train_inputs, test_inputs)

        # set defaults
        self.set_defaults()

        # initialize algorithm specifics
        self.lah = True
        self.lm = False
        self.loss_method = loss_method
        self.regression = regression
        self.initialize_algo(dict)

        # post init changes
        # self.post_init_changes()

        # optimal solutions (not needed as input)
        self.setup_optimal_solutions(z_stars_train, z_stars_test, x_stars_train, x_stars_test, 
                                     y_stars_train, y_stars_test)

        # create_all_loss_fns
        self.create_all_loss_fns(loss_method, regression)
        

        self.step_varying_num = step_varying_num # 50

        # neural network setup
        self.initialize_neural_network(nn_cfg, plateau_decay)

        # init to track training
        self.init_train_tracking()

        self.compute_avg_opt()

        
    def compute_avg_opt(self):
        pass

    def reinit_losses(self):
        self.create_all_loss_fns(self.loss_method, self.regression)


    def set_defaults(self):
        # unless turned off in the subclass, these are the default settings
        self.factors_required = False
        self.factor_static = None


    def initialize_essentials(self, jit, eval_unrolls, train_unrolls, train_inputs, test_inputs):
        self.jit = jit
        self.eval_unrolls = eval_unrolls
        self.train_unrolls = train_unrolls
        self.train_inputs, self.test_inputs = train_inputs, test_inputs
        self.N_train, self.N_test = self.train_inputs.shape[0], self.test_inputs.shape[0]
        self.static_flag = True


    def setup_optimal_solutions(self, z_stars_train, z_stars_test, x_stars_train=None, 
                                x_stars_test=None, y_stars_train=None, y_stars_test=None):
        if z_stars_train is not None:
            self.z_stars_train = jnp.array(z_stars_train) # jnp.array(dict['z_stars_train'])
            self.z_stars_test = jnp.array(z_stars_test) # jnp.array(dict['z_stars_test'])
        else:
            self.z_stars_train, self.z_stars_test = None, None

    def transform_params(self, params, n_iters):
        # transformed_params = jnp.exp(params[0][:n_iters - 1, 0])
        transformed_params = jnp.exp(params[0][:n_iters, :])
        return transformed_params

    def create_end2end_loss_fn(self, bypass_nn, diff_required):
        supervised = self.supervised and diff_required
        loss_method = self.loss_method

        def predict(params, input, q, iters, z_star, key, factor):
            if self.algo == 'scs':
                q = lin_sys_solve(factor, q)
            else:
                pass
            z0 = self.predict_warm_start(params, input, key, bypass_nn)

            if self.train_fn is not None:
                train_fn = self.train_fn
            else:
                train_fn = self.k_steps_train_fn
            if self.eval_fn is not None:
                eval_fn = self.eval_fn
            else:
                eval_fn = self.k_steps_eval_fn

            if diff_required:
                if self.factors_required:
                    z_final, iter_losses = train_fn(k=iters,
                                                    z0=z0,
                                                    q=q,
                                                    supervised=supervised,
                                                    z_star=z_star,
                                                    factor=factor)
                else:
                    z_final, iter_losses = train_fn(k=iters,
                                                    z0=z0,
                                                    q=q,
                                                    supervised=supervised,
                                                    z_star=z_star)
            else:
                if self.factors_required:
                    eval_out = eval_fn(k=iters,
                                       z0=z0,
                                       q=q,
                                       factor=factor,
                                       supervised=supervised,
                                       z_star=z_star)
                else:
                    eval_out = eval_fn(k=iters,
                                       z0=z0,
                                       q=q,
                                       supervised=supervised,
                                       z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]

                angles = None
            print('z_final', z_final)
            loss = self.final_loss(loss_method, z_final, iter_losses, supervised, z0, z_star)

            # penalty_loss = calculate_total_penalty(self.N_train, params, self.b, self.c, self.delta)
            # penalty_loss = calculate_pinsker_penalty(self.N_train, params, self.b, self.c, self.delta)
            # loss = loss #+ self.penalty_coeff * penalty_loss

            if diff_required:
                return loss
            else:
                return_out = (loss, iter_losses, z_all_plus_1, angles) + eval_out[3:]
                return return_out
        loss_fn = self.predict_2_loss(predict, diff_required)
        return loss_fn
    

    def train_stochastic(self, prev_params, train_case, placeholder, n_iters, params, state):
        mean, var = self.gauss_mean, self.gauss_var
        if train_case == 'stochastic_one_step_grad':
            alpha = one_step_stochastic_quad_gd_solver(mean, var, prev_params, self.P)
            params[0] = jnp.log(jnp.array([[alpha]]))
        elif train_case == 'stochastic_two_step_quad': 
            # gradients = self.compute_gradients(batch_inputs, batch_q_data)
            P = self.P
            alpha, beta = two_step_stochastic_quad_gd_solver(mean, var, prev_params, P)
            # import pdb
            # pdb.set_trace()
            params[0] = jnp.log(jnp.array([[alpha, beta]])).T
        elif train_case == 'stochastic_three_step_quad': 
            # gradients = self.compute_gradients(batch_inputs, batch_q_data)
            P = self.P
            alpha, beta, gamma = three_step_stochastic_quad_gd_solver(mean, var, prev_params, P)
            # import pdb
            # pdb.set_trace()
            params[0] = jnp.log(jnp.array([[alpha, beta, gamma]])).T
            # return state.value, jnp.log(jnp.array([[alpha, beta, gamma]])).T, state
        elif train_case == 'stochastic_multi_step_quad': 
            params_matrix = jnp.tile(prev_params[0], (placeholder.shape[0], 1))
            # import pdb
            # pdb.set_trace()
            z_bar = stochastic_get_z_bar(self.gauss_mean, self.gauss_var, prev_params, self.P)
            z_bar_mat = jnp.tile(z_bar, (placeholder.shape[0], 1))
            results = self.optimizer.update(params=params,
                                            state=state,
                                            inputs=z_bar_mat,
                                            b=placeholder,
                                            iters=self.train_unrolls,
                                            z_stars=placeholder,
                                            key=n_iters)
            params, state = results
        # return params
        # import pdb
        # pdb.set_trace()
        return state.value, params, state


    def train_batch(self, batch_indices, inputs, params, state, n_iters, train_case='gradient'):
        if train_case[:10] == 'stochastic':
            # prev_params = [inputs[0]]
            prev_params = inputs
            placeholder = self.q_mat_train
            # params[0] = self.train_stochastic(prev_params, train_case, params, state)
            # return state.value, params, state
            return self.train_stochastic(prev_params, train_case, placeholder, n_iters, params, state)
        batch_inputs = inputs[batch_indices, :]
        batch_q_data = self.q_mat_train[batch_indices, :]
        batch_z_stars = self.z_stars_train[batch_indices, :]

        key = n_iters #1 if params[0].shape[0] == 0 else self.train_unrolls #state.iter_num

        if train_case == 'one_step_grad':
            gradients = self.compute_gradients(batch_inputs, batch_q_data)
            alpha = one_step_gd_solver(batch_z_stars, batch_inputs, gradients)
            params[0] = jnp.log(jnp.array([[alpha]]))
        elif train_case == 'two_step_quad': 
            # gradients = self.compute_gradients(batch_inputs, batch_q_data)
            P = self.P
            alpha, beta = two_step_quad_gd_solver(batch_z_stars, batch_inputs, P)
            # import pdb
            # pdb.set_trace()
            params[0] = jnp.log(jnp.array([[alpha, beta]])).T
        elif train_case == 'three_step_quad': 
            # gradients = self.compute_gradients(batch_inputs, batch_q_data)
            P = self.P
            alpha, beta, gamma = three_step_quad_gd_solver(batch_z_stars, batch_inputs, P)
            # import pdb
            # pdb.set_trace()
            params[0] = jnp.log(jnp.array([[alpha, beta, gamma]])).T
        else:
            # gradient-based methods
            results = self.optimizer.update(params=params,
                                            state=state,
                                            inputs=batch_inputs,
                                            b=batch_q_data,
                                            iters=self.train_unrolls,
                                            z_stars=batch_z_stars,
                                            key=key)
            params, state = results
        self.key = key
        print('params', params)
        return state.value, params, state

    def evaluate(self, k, inputs, b, z_stars, fixed_ws, key, factors=None, tag='test', light=False):
        if self.factors_required and not self.factor_static_bool:
            return self.dynamic_eval(k, inputs, b, z_stars, 
                                     factors=factors, key=self.key, tag=tag, fixed_ws=fixed_ws)
        else:
            return self.static_eval(k, inputs, b, z_stars, key, tag=tag, 
                                    fixed_ws=fixed_ws, light=light)

    def short_test_eval(self):
        z_stars_test = self.z_stars_test

        if self.lah:
            z0_inits = z_stars_test * 0
            if self.algo == 'lah_scs':
                z0_inits = jnp.hstack([z0_inits, jnp.ones((z0_inits.shape[0], 1))])
            elif self.algo == 'lah_osqp':
                z0_inits = z0_inits[:, :self.m + self.n]
        else:
            z0_inits = self.test_inputs
        test_loss, test_out, time_per_prob = self.static_eval(self.train_unrolls,
                                                            z0_inits,
                                                                self.q_mat_test,
                                                                z_stars_test,
                                                                self.train_unrolls)

        self.te_losses.append(test_loss)

        time_per_iter = time_per_prob / self.train_unrolls
        return test_loss, time_per_iter

    def dynamic_eval(self, k, inputs, b, z_stars, factors, key, tag='test', fixed_ws=False):
        if fixed_ws:
            curr_loss_fn = self.loss_fn_fixed_ws
        else:
            curr_loss_fn = self.loss_fn_eval
        num_probs, _ = inputs.shape

        test_time0 = time.time()

        loss, out = curr_loss_fn(self.params, inputs, b, k, z_stars, key, factors)
        time_per_prob = (time.time() - test_time0)/num_probs

        return loss, out, time_per_prob

    def static_eval(self, k, inputs, b, z_stars, key, tag='test', fixed_ws=False, light=False):
        curr_loss_fn = self.loss_fn_fixed_ws if fixed_ws else self.loss_fn_eval
        if tag == 'silver':
            curr_loss_fn = self.loss_fn_eval_silver
        elif tag == 'conj_grad':
            curr_loss_fn = self.loss_fn_eval_conj_grad
        num_probs, _ = inputs.shape
        test_time0 = time.time()
        loss, out = curr_loss_fn(self.params, inputs, b, k, z_stars, key)
        time_per_prob = (time.time() - test_time0)/num_probs
        return loss, out, time_per_prob


    def initialize_neural_network(self, nn_cfg, plateau_decay, alista_cfg=None):
        # neural network
        self.epochs, self.lr = nn_cfg.get('epochs', 10), nn_cfg.get('lr', 1e-3)

        # batching
        batch_size = nn_cfg.get('batch_size', self.N_train)
        self.batch_size = min([batch_size, self.N_train])
        self.num_batches = int(self.N_train/self.batch_size)

        # layer sizes
        input_size = self.train_inputs.shape[1]

        output_size = self.output_size
        hidden_layer_sizes = nn_cfg.get('intermediate_layer_sizes', [])

        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        self.layer_sizes = layer_sizes

        self.init_params()

        self.optimizer_method = nn_cfg.get('method', 'adam')
        self.init_optimizer()


    def init_optimizer(self):
        # initializes the optimizer
        if self.optimizer_method == 'adam':
            self.optimizer = OptaxSolver(opt=optax.adam(
                self.lr), fun=self.loss_fn_train, has_aux=False)
        elif self.optimizer_method == 'sgd':
            self.optimizer = OptaxSolver(opt=optax.sgd(
                self.lr), fun=self.loss_fn_train, has_aux=False)

        # Initialize state with first elements of training data as inputs
        batch_indices = np.arange(5) #jnp.arange(self.N_train)
        if self.lah:
            input_init = self.lah_train_inputs[batch_indices, :] #self.train_inputs[batch_indices, :]
        # elif self.lm:
        #     input_init = 0 * self.z_stars_train[batch_indices, :]
        else:
            input_init = self.train_inputs[batch_indices, :]
        q_init = self.q_mat_train[batch_indices, :]
        z_stars_init = self.z_stars_train[batch_indices, :] #if self.supervised else None

        if not hasattr(self, 'num_const_steps'):
            self.num_const_steps = 1

        if self.lah:
            self.state = self.optimizer.init_state(init_params=[self.params[0][:int(self.train_unrolls / self.num_const_steps), :]],
                                                   inputs=input_init,
                                                   b=q_init,
                                                   iters=self.train_unrolls,
                                                   z_stars=z_stars_init,
                                                   key=self.train_unrolls)
        else:
            self.state = self.optimizer.init_state(init_params=self.params,
                                                   inputs=input_init,
                                                   b=q_init,
                                                   iters=self.train_unrolls,
                                                   z_stars=z_stars_init,
                                                   key=self.train_unrolls)

            
    def init_params(self):
        # initialize weights of neural network
        self.mean_params = init_network_params(self.layer_sizes, random.PRNGKey(0))

        # # initialize the stddev
        # init_stddev_var = 1e-6
        # self.sigma_params = init_variance_network_params(self.layer_sizes, self.init_var, 
        #                                                  random.PRNGKey(1), 
        #                                                 init_stddev_var)
        
        # # initialize the prior
        # self.prior_param = jnp.log(self.init_var) * jnp.ones(2 * len(self.layer_sizes))

        self.params = [self.mean_params] #, self.sigma_params, self.prior_param]


    def create_all_loss_fns(self, loss_method, supervised):
        # to describe the final loss function (not the end-to-end loss fn)
        self.loss_method = loss_method
        self.supervised = supervised

        self.train_fn = self.k_steps_train_fn
        self.eval_fn = self.k_steps_eval_fn

        e2e_loss_fn = self.create_end2end_loss_fn

        # end-to-end loss fn for training
        self.loss_fn_train = e2e_loss_fn(bypass_nn=False, diff_required=True)

        # end-to-end loss fn for evaluation
        self.loss_fn_eval = e2e_loss_fn(bypass_nn=False, diff_required=False)

        # end-to-end added fixed warm start eval - bypasses neural network
        self.loss_fn_fixed_ws = e2e_loss_fn(bypass_nn=True, diff_required=False)


    def init_train_tracking(self):
        self.epoch = 0
        self.tr_losses = None
        self.te_losses = None
        self.train_data = []
        self.tr_losses_batch = []
        self.te_losses = []


    def predict_warm_start(self, params, input, key, bypass_nn):
        """
        gets the warm-start
        bypass_nn means we ignore the neural network and set z0=input
        """
        if bypass_nn:
            z0 = input
        else:
            # old stochastic
            # perturb = get_perturbed_weights(random.PRNGKey(key), self.layer_sizes, jnp.sqrt(sigma))
            # perturbed_weights = [(perturb[i][0] + params[i][0], 
            #                       0*perturb[i][1] + params[i][1]) for i in range(len(params))]
            # print('perturbed_weights', perturbed_weights)

            # new stochastic
            mean_params = params[0]
            nn_output = predict_y(mean_params, input)

            # mean_params, sigma_params, prior_var = params[0], params[1], params[2]
            # if self.deterministic:
            #     nn_output = predict_y(mean_params, input)
            # else:
            #     perturb = get_perturbed_weights(random.PRNGKey(key), self.layer_sizes, 1)
            #     perturbed_weights = [(perturb[i][0] * jnp.sqrt(jnp.exp(sigma_params[i][0])) + mean_params[i][0], 
            #                         perturb[i][1] * jnp.sqrt(jnp.exp(sigma_params[i][1])) + mean_params[i][1]) for i in range(len(mean_params))]
            #     # perturbed_weights = [(perturb[i][0] * jnp.sqrt(1 / (1 + jnp.exp(-sigma_params[i][0]))) + mean_params[i][0], 
            #     #                     perturb[i][1] * jnp.sqrt(1 / (1 + jnp.exp(-sigma_params[i][1]))) + mean_params[i][1]) for i in range(len(mean_params))]

            #     nn_output = predict_y(perturbed_weights, input)

            # deterministic
            # nn_output = predict_y(params, input)
            z0 = nn_output
        if self.algo == 'scs':
            z0_full = jnp.ones(z0.size + 1)
            z0_full = z0_full.at[:z0.size].set(z0)
        else:
            z0_full = z0
        return z0_full


    def final_loss(self, loss_method, z_last, iter_losses, supervised, z0, z_star):
        """
        encodes several possible loss functions

        z_last is the last iterate from DR splitting
        z_penultimate is the second to last iterate from DR splitting

        z_star is only used if supervised

        z0 is only used if the loss_method is first_2_last
        """
        # return iter_losses[-1]
        if supervised:
            if loss_method == 'constant_sum':
                loss = iter_losses[1:].sum()
            elif loss_method == 'fixed_k':
                # loss = jnp.linalg.norm(z_last[:-1]/z_star[-1] - z_star)
                loss = iter_losses[-1]
        else:
            if loss_method == 'increasing_sum':
                weights = (1+jnp.arange(iter_losses.size))
                loss = iter_losses @ weights
            elif loss_method == 'constant_sum':
                loss = iter_losses[1:].sum()
            elif loss_method == 'fixed_k':
                loss = iter_losses[-1]
            elif loss_method == 'first_2_last':
                loss = jnp.linalg.norm(z_last - z0)
        return loss

    def get_out_axes_shape(self, diff_required):
        if diff_required:
            # out_axes for (loss)
            out_axes = (0)
        else:
            # out_axes for
            #   (loss, iter_losses, angles, primal_residuals, dual_residuals, out)
            #   out = (all_z_, z_next, alpha, all_u, all_v)
            # out_axes = (0, 0, 0, 0)
            # if self.out_axes_length is None:
            if hasattr(self, 'out_axes_length'):
                out_axes = (0,) * self.out_axes_length
            else:
                out_axes = (0,) * 4
        return out_axes

    def predict_2_loss(self, predict, diff_required):
        out_axes = self.get_out_axes_shape(diff_required)

        # just for reference, the arguments for predict are
        #   predict(params, input, q, iters, z_star, factor)

        # for either of the following cases
        #   1. no factors are needed (pass in None as a static argument)
        #   2. factor is constant for all problems (pass in the same factor as static argument)
        predict_partial = partial(predict, factor=self.factor_static)

        batch_predict = vmap(predict_partial,
                                in_axes=(None, 0, 0, None, 0, None),
                                out_axes=out_axes)
        

        # @partial(jit, static_argnums=(3, 5,))
        @partial(jit, static_argnames=['iters', 'key'])
        def loss_fn(params, inputs, b, iters, z_stars, key):
            if diff_required:
                losses = batch_predict(params, inputs, b, iters, z_stars, key)
                return losses.mean()
            
                # return losses.mean()
                q = losses.mean() / self.penalty_coeff

                penalty_loss = self.calculate_total_penalty(self.N_train, params, self.b, 
                                                            self.c, 
                                                            self.delta)

                bisec = Bisection(optimality_fun=kl_inv_fn, lower=0.0, upper=1.0, 
                                    check_bracket=False,
                                    jit=True)
                if self.algo == 'lista':
                    factor = 0.01
                else:
                    factor = 0.1
                q_expit = 1 / (1 + jnp.exp(-factor * (q - 0)))

                out = bisec.run(q=q_expit, c=penalty_loss)
                r = out.params
                p = (1 - q_expit) * r + q_expit

                if self.deterministic:
                    return q_expit
                return p + 1000 * (penalty_loss - self.target_pen) ** 2
                # return q #+ jnp.sqrt(penalty_loss / 2) + 100 * (penalty_loss - self.target_pen) ** 2
            else:
                predict_out = batch_predict(
                    params, inputs, b, iters, z_stars, key)
                losses = predict_out[0]
                return losses.mean(), predict_out

        return loss_fn


    # def calculate_total_penalty(self, N_train, params, c, b, delta):
    #     pi_pen = jnp.log(jnp.pi ** 2 * N_train / (6 * delta))
    #     # log_pen = 2 * jnp.log(b * jnp.log(c / jnp.exp(params[2])))
    #     log_pen = 2 * jnp.log(b * jnp.log(c / jnp.exp(params[2][0])))
    #     # import pdb
    #     # pdb.set_trace()
    #     penalty_loss = self.compute_all_params_KL(params[0], params[1], 
    #                                         params[2]) + pi_pen + log_pen
    #     return penalty_loss /  N_train

    def round_priors(self, priors, lambda_max, b):
        lambd = jnp.clip(jnp.exp(priors), a_max=lambda_max)
        a = jnp.round(b * jnp.log((lambda_max + 1e-6) / lambd))
        rounded_lambd = lambda_max * jnp.exp(-a / b)
        return jnp.log(rounded_lambd)
        # lambd = lambda_max / (1 + jnp.exp(-priors))
        # a = jnp.round(b * jnp.log(lambda_max / lambd))
        # rounded_lambd = lambda_max * jnp.exp(-a / b)
        # # return jnp.log(rounded_priors)
        # return jnp.log(rounded_lambd / (lambda_max - rounded_lambd))
        # return rounded_lambd
    

    def calculate_total_penalty(self, N_train, params, c, b, delta, prior=0):
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


    def compute_all_params_KL(self, mean_params, sigma_params, eta):
        return 0
        lambda_max = self.c
        total_pen = 0
        for i, params in enumerate(mean_params):
            weight_matrix, bias_vector = params
            weight_sigma, bias_sigma = sigma_params[i][0], sigma_params[i][1]
            # curr_lambd_weight = lambda_max / (1 + jnp.exp(-eta[2*i]))
            curr_lambd_weight = jnp.exp(eta[2*i])
            total_pen += compute_single_param_KL(weight_matrix, 
                                                 jnp.exp(weight_sigma), curr_lambd_weight)
            # curr_lambd_bias = lambda_max / (1 + jnp.exp(-eta[2*i+1]))
            curr_lambd_bias = jnp.exp(eta[2*i+1])
            total_pen += compute_single_param_KL(bias_vector, 
                                                 jnp.exp(bias_sigma), curr_lambd_bias)
        return total_pen


    def compute_weight_norm_squared(self, nn_params):
        return 0, 0
        weight_norms = np.zeros(len(nn_params))
        nn_weights = nn_params
        num_weights = 0
        for i, params in enumerate(nn_weights):
            weight_matrix, bias_vector = params
            weight_norms[i] = jnp.linalg.norm(weight_matrix) ** 2 + jnp.linalg.norm(bias_vector) ** 2
            num_weights += weight_matrix.size + bias_vector.size
        return weight_norms.sum(), num_weights

    
    def calculate_avg_posterior_var(self, params):
        return 0, 0
        sigma_params = params[1]
        flattened_params = jnp.concatenate([jnp.ravel(weight_matrix) for weight_matrix, _ in sigma_params] + 
                                        [jnp.ravel(bias_vector) for _, bias_vector in sigma_params])
        variances = jnp.exp(flattened_params)
        # flattened_params = jnp.concatenate([jnp.ravel(weight_matrix) for weight_matrix, _ in sigma_params] + 
        #                                 [jnp.ravel(bias_vector) for _, bias_vector in sigma_params])
        # variances = jnp.exp(flattened_params)
        avg_posterior_var = variances.mean()
        stddev_posterior_var = variances.std()
        return avg_posterior_var, stddev_posterior_var