from functools import partial

import cvxpy as cp
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scs
from scipy.sparse import csc_matrix

from lasco.algo_steps import (
    create_M,
    get_scaled_vec_and_factor,
    k_steps_eval_lasco_scs,
    k_steps_train_lasco_scs,
    lin_sys_solve,
)
from lasco.l2ws_model import L2WSmodel


class LASCOSCSmodel(L2WSmodel):
    def __init__(self, **kwargs):
        super(LASCOSCSmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        """
        the input_dict is required to contain these keys
        otherwise there is an error
        """
        self.factors_required = True
        self.factor_static_bool = input_dict.get('factor_static_bool', True)
        self.algo = 'lasco_scs'
        self.factors_required = True
        self.hsde = input_dict.get('hsde', True)
        self.m, self.n = input_dict['m'], input_dict['n']
        self.cones = input_dict['cones']
        self.proj, self.static_flag = input_dict['proj'], input_dict.get('static_flag', True)
        self.q_mat_train, self.q_mat_test = input_dict['q_mat_train'], input_dict['q_mat_test']

        M = input_dict['static_M']
        self.P = M[:self.n, :self.n]
        self.A = -M[self.n:, :self.n]
        self.M = M

        # factor = input_dict['static_algo_factor']
        # self.factor = factor
        # self.factor_static = factor

        # hyperparameters of scs
        # self.rho_x = input_dict.get('rho_x', 1)
        # self.scale = input_dict.get('scale', 1)
        # self.alpha_relax = input_dict.get('alpha_relax', 1)

        # not a hyperparameter, but used for scale knob
        self.zero_cone_size = self.cones['z'] #input_dict['zero_cone_size']
        lightweight = input_dict.get('lightweight', False)

        self.output_size = self.n + self.m
        self.out_axes_length = 8

        # custom_loss
        custom_loss = input_dict.get('custom_loss')

        self.k_steps_train_fn = partial(k_steps_train_lasco_scs, proj=self.proj,
                                        jit=self.jit,
                                        # m=self.m,
                                        # n=self.n,
                                        # zero_cone_size=self.zero_cone_size,
                                        hsde=True)
        self.k_steps_eval_fn = partial(k_steps_eval_lasco_scs, proj=self.proj,
                                       P=self.P, A=self.A,
                                       zero_cone_size=self.zero_cone_size,
                                       jit=self.jit,
                                       hsde=True,
                                       custom_loss=custom_loss,
                                       lightweight=lightweight)
        
    def init_params(self):
        # p = jnp.diag(self.P)
        self.mean_params = 0*jnp.ones((self.train_unrolls, 4))

        self.sigma_params = -jnp.ones((self.train_unrolls, 4)) * 10

        # initialize the prior
        self.prior_param = jnp.log(self.init_var) * jnp.ones(2)

        self.params = [self.mean_params, self.sigma_params, self.prior_param]


    # def setup_optimal_solutions(self, dict):
    def setup_optimal_solutions(self, 
                                z_stars_train, 
                                z_stars_test, 
                                x_stars_train=None, 
                                x_stars_test=None, 
                                y_stars_train=None, 
                                y_stars_test=None):
        # if dict.get('x_stars_train', None) is not None:
        if x_stars_train is not None:
            # self.y_stars_train, self.y_stars_test = dict['y_stars_train'], dict['y_stars_test']
            # self.x_stars_train, self.x_stars_test = dict['x_stars_train'], dict['x_stars_test']
            # self.z_stars_train = jnp.array(dict['z_stars_train'])
            # self.z_stars_test = jnp.array(dict['z_stars_test'])
            self.z_stars_train = jnp.array(z_stars_train)
            self.z_stars_test = jnp.array(z_stars_test)
            self.x_stars_train = jnp.array(x_stars_train)
            self.x_stars_test = jnp.array(x_stars_test)
            self.y_stars_train = jnp.array(y_stars_train)
            self.y_stars_test = jnp.array(y_stars_test)
            self.u_stars_train = jnp.hstack([self.x_stars_train, self.y_stars_train])
            self.u_stars_test = jnp.hstack([self.x_stars_test, self.y_stars_test])
        if z_stars_train is not None:
            self.z_stars_train = z_stars_train
            self.z_stars_test = z_stars_test
        else:
            self.z_stars_train, self.z_stars_test = None, None

    def create_end2end_loss_fn(self, bypass_nn, diff_required):
        supervised = self.supervised and diff_required
        loss_method = self.loss_method

        def predict(params, input, q, iters, z_star, key, factor):
            # q = lin_sys_solve(factor, q)
            z0 = jnp.zeros(z_star.size + 1) #self.predict_warm_start(params, input, key, bypass_nn)
            z0 = z0.at[-1].set(1)

            if self.train_fn is not None:
                train_fn = self.train_fn
            else:
                train_fn = self.k_steps_train_fn
            if self.eval_fn is not None:
                eval_fn = self.eval_fn
            else:
                eval_fn = self.k_steps_eval_fn

            # do all of the factorizations
            factors1 = jnp.zeros((self.train_unrolls, self.m + self.n, self.m + self.n))
            factors2 = jnp.zeros((self.train_unrolls, self.m + self.n), dtype=jnp.int32)
            scaled_vecs = jnp.zeros((self.train_unrolls, self.m + self.n))

            rho_xs, rho_ys = params[0][:, 0], params[0][:, 1]
            for i in range(self.train_unrolls):
                rho_x, rho_y = jnp.exp(rho_xs[i]), jnp.exp(rho_ys[i])
                scale = 1 / rho_y
                factor, scale_vec = get_scaled_vec_and_factor(self.M, rho_x, scale, 
                                                              self.m, self.n, 
                                                              zero_cone_size=0, hsde=True)
                
                factors1 = factors1.at[i, :, :].set(factor[0])
                factors2 = factors2.at[i, :].set(factor[1])
                scaled_vecs = scaled_vecs.at[i, :].set(scale_vec)


            all_factors = factors1, factors2
            scs_params = (params[0], all_factors, scaled_vecs)

            if diff_required:
                z_final, iter_losses = train_fn(k=iters,
                                                z0=z0,
                                                q=q,
                                                params=scs_params,
                                                supervised=supervised,
                                                z_star=z_star) #,
                                                #factor=factor)
            else:
                eval_out = eval_fn(k=iters,
                                    z0=z0,
                                    q=q,
                                    params=scs_params,
                                    supervised=supervised,
                                    z_star=z_star)
                z_final, iter_losses, z_all_plus_1 = eval_out[0], eval_out[1], eval_out[2]

                angles = None

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


    def solve_c(self, z0_mat, q_mat, rel_tol, abs_tol, max_iter=10000):
        # assume M doesn't change across problems
        # static problem data
        m, n = self.m, self.n
        P, A = self.P, self.A

        # set the solver
        b_zeros, c_zeros = np.zeros(m), np.zeros(n)

        # osqp_solver = osqp.OSQP()
        P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))
        c_data = dict(P=P_sparse, A=A_sparse, c=c_zeros, b=b_zeros)
        
        solver = scs.SCS(c_data,
                         self.cones,
                         normalize=False,
                         scale=self.scale,
                         adaptive_scale=False,
                         rho_x=self.rho_x,
                         alpha=self.alpha_relax,
                         acceleration_lookback=0,
                         max_iters=max_iter,
                         eps_abs=abs_tol,
                         eps_rel=rel_tol,
                         verbose=False)

        

        # q = q_mat[0, :]
        # c, l, u = np.zeros(n), np.zeros(m), np.zeros(m)
        # osqp_solver.setup(P=P_sparse, q=c, A=A_sparse, l=l, u=u, alpha=1, 
        #                   rho=1, sigma=1, polish=False,
        #                   adaptive_rho=False, scaling=0, max_iter=max_iter, 
        #                   verbose=True, eps_abs=abs_tol, eps_rel=rel_tol)

        num = z0_mat.shape[0]
        solve_times = np.zeros(num)
        solve_iters = np.zeros(num)
        x_sols = jnp.zeros((num, n))
        y_sols = jnp.zeros((num, m))
        for i in range(num):
            # set c, l, u
            # c, l, u = q_mat[i, :n], q_mat[i, n:n + m], q_mat[i, n + m:]
            # osqp_solver.update(q=np.array(c))
            # osqp_solver.update(l=np.array(l), u=np.array(u))
            b, c = q_mat[i, n:], q_mat[i, :n]
            solver.update(b=np.array(b))
            solver.update(c=np.array(c))

            # set the warm start
            x, y, s = self.get_xys_from_z(z0_mat[i, :], m, n)
            x_ws, y_ws = np.array(x), np.array(y)
            s_ws = np.array(s)

            # fix warm start
            # osqp_solver.warm_start(x=x_ws, y=y_ws)
            sol = solver.solve(warm_start=True, x=x_ws, y=y_ws, s=s_ws)

            # solve
            # results = osqp_solver.solve()
            # sol = solver.solve(warm_start=True, x=np.array(x), y=np.array(y), s=np.array(s))

            # set the solve time in seconds
            # solve_times[i] = results.info.solve_time
            # solve_iters[i] = results.info.iter
            solve_times[i] = sol['info']['solve_time'] #/ 1000
            solve_iters[i] = sol['info']['iter']

            # set the results
            x_sols = x_sols.at[i, :].set(sol['x'])
            y_sols = y_sols.at[i, :].set(sol['y'])

        return solve_times, solve_iters, x_sols, y_sols
    
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
    

def get_scs_factor(P, A, cones, rho_x=1, scale=1):
    m, n = A.shape
    zero_cone_size = cones['z']
    P_jax = jnp.array(P.todense())
    A_jax = jnp.array(A.todense())
    M_jax = create_M(P_jax, A_jax)
    algo_factor, scale_vec = get_scaled_vec_and_factor(M_jax, rho_x, scale, m, n,
                                                        zero_cone_size)
    return M_jax, algo_factor, scale_vec


def solve_cvxpy_get_params(prob, cp_param, theta_values):
    cp_param.value = theta_values[0]
    prob.solve()
    data, _, __ = prob.get_problem_data(cp.SCS)
    c, b = data['c'], data['b']
    A = data['A']

    m = b.size
    n = c.size
    P = csc_matrix(np.zeros((n, n)))

    cones_cp = data['dims']
    cones = {'z': cones_cp.zero, 'l': cones_cp.nonneg, 'q': cones_cp.soc, 's': cones_cp.psd}

    N = len(theta_values)

    q_mat = np.zeros((N, m + n))
    z_stars = np.zeros((N, m + n))
    x_stars = np.zeros((N, n))
    y_stars = np.zeros((N, m))
    for i in range(N):
        cp_param.value = theta_values[i]
        prob.solve()
        data, _, __ = prob.get_problem_data(cp.SCS)
        c, b = data['c'], data['b']

        # q = (c, b)
        q_mat[i, :n] = c
        q_mat[i, n:] = b

        # get the optimal solution
        x_star = prob.solution.attr['solver_specific_stats']['x']
        y_star = prob.solution.attr['solver_specific_stats']['y']
        s_star = prob.solution.attr['solver_specific_stats']['s']

        # transform the solution to the z variable
        z_star = np.concatenate([x_star, y_star + s_star])
        z_stars[i, :] = z_star
        x_stars[i, :] = x_star
        y_stars[i, :] = y_star
    return z_stars, q_mat, cones, P, A
