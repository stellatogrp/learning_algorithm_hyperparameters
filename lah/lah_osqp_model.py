from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import osqp
from scipy.sparse import csc_matrix

from lah.algo_steps_osqp import k_steps_eval_lah_osqp, k_steps_train_lah_osqp, unvec_symm
from lah.l2ws_model import L2WSmodel


class LAHOSQPmodel(L2WSmodel):
    def __init__(self, **kwargs):
        super(LAHOSQPmodel, self).__init__(**kwargs)

    def initialize_algo(self, input_dict):
        # self.m, self.n = self.A.shape
        self.algo = 'lah_osqp'
        self.m, self.n = input_dict['m'], input_dict['n']
        self.q_mat_train, self.q_mat_test = input_dict['q_mat_train'], input_dict['q_mat_test']

        self.rho = input_dict['rho']
        self.sigma = input_dict.get('sigma', 1)
        self.alpha = input_dict.get('alpha', 1)
        self.output_size = self.n + self.m

        # custom_loss
        custom_loss = input_dict.get('custom_loss')

        self.num_const_steps = input_dict.get('num_const_steps')
        self.idx_mapping = jnp.arange(self.eval_unrolls) // self.num_const_steps


        """
        break into the 2 cases
        1. factors are the same for each problem (i.e. matrices A and P don't change)
        2. factors change for each problem
        """
        self.factors_required = True
        self.factor_static_bool = input_dict.get('factor_static_bool', True)
        if self.factor_static_bool:
            self.A = input_dict['A']
            self.P = input_dict['P']
            # self.M = self.P + sigma * jnp.eye(n) + A.T @ jnp.diag(rho_vec) @ A
            self.factor_static = input_dict['factor']
            self.k_steps_train_fn = partial(
                k_steps_train_lah_osqp, A=self.A, idx_mapping=self.idx_mapping, jit=self.jit)
            self.k_steps_eval_fn = partial(k_steps_eval_lah_osqp, P=self.P,
                                           A=self.A, 
                                           idx_mapping=self.idx_mapping,
                                           jit=self.jit, 
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
        N = self.q_mat_train.shape[0]
        self.lah_train_inputs = jnp.zeros((N, self.n + self.m))


    def init_params(self):
        # init step-varying params
        step_varying_params = jnp.ones((self.step_varying_num, 4))

        # init steady_state_params
        steady_state_params = jnp.ones((1, 4))

        self.params = [jnp.vstack([step_varying_params, steady_state_params])]
        # self.mean_params = jnp.ones((self.train_unrolls, 3))

        # self.sigma_params = -jnp.ones((self.train_unrolls, 3)) * 10

        # # initialize the prior
        # self.prior_param = jnp.log(self.init_var) * jnp.ones(2)

        # self.params = [self.mean_params, self.sigma_params, self.prior_param]


    def create_end2end_loss_fn(self, bypass_nn, diff_required):
        supervised = self.supervised and diff_required
        loss_method = self.loss_method

        def predict(params, input, q, iters, z_star, key, factor):
            # z0 = jnp.zeros(self.m + self.n) #self.predict_warm_start(params, input, key, bypass_nn)
            z0 = input

            if diff_required:
                n_iters = key #self.train_unrolls if key else 1
            else:
                n_iters = min(iters, 51)

            if self.train_fn is not None:
                train_fn = self.train_fn
            else:
                train_fn = self.k_steps_train_fn
            if self.eval_fn is not None:
                eval_fn = self.eval_fn
            else:
                eval_fn = self.k_steps_eval_fn

            # do all of the factorizations
            factors1 = jnp.zeros((n_iters, self.n, self.n))
            factors2 = jnp.zeros((n_iters, self.n), dtype=jnp.int32)
            rhos, sigmas = params[0][:, 0], params[0][:, 1]
            for i in range(n_iters):
                rho, sigma = jnp.exp(rhos[i]), jnp.exp(sigmas[i])
                rho_vec = rho * jnp.ones(self.m)
                M = self.P + sigma * jnp.eye(self.n) + self.A.T @ jnp.diag(rho_vec) @ self.A
                factor = jsp.linalg.lu_factor(M)
                
                factors1 = factors1.at[i, :, :].set(factor[0])
                factors2 = factors2.at[i, :].set(factor[1])
                
            all_factors = factors1, factors2
            osqp_params = (params[0], all_factors)


            if diff_required:
                z_final, iter_losses = train_fn(k=iters,
                                                z0=z0,
                                                q=q,
                                                params=osqp_params,
                                                supervised=supervised,
                                                z_star=z_star) #,
                                                #factor=factor)
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

            # penalty_loss = calculate_total_penalty(self.N_train, params, self.b, self.c, self.delta)
            # penalty_loss = calculate_pinsker_penalty(self.N_train, params, self.b, self.c, self.delta)
            loss = loss #+ self.penalty_coeff * penalty_loss

            if diff_required:
                return loss
            else:
                return_out = (loss, iter_losses, z_all_plus_1, angles) + eval_out[3:]
                return return_out
        loss_fn = self.predict_2_loss(predict, diff_required)
        return loss_fn


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

    def solve_c(self, z0_mat, q_mat, rel_tol, abs_tol, max_iter=40000):
        # assume M doesn't change across problems
        # static problem data
        m, n = self.m, self.n
        nc2 = int(n * (n + 1) / 2)

        if self.factor_static_bool:
            P, A = self.P, self.A
        else:
            P, A = np.ones((n, n)), np.zeros((m, n))
        P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))


        osqp_solver = osqp.OSQP()
        

        # q = q_mat[0, :]
        c, l, u = np.zeros(n), np.zeros(m), np.zeros(m)  # noqa
        
        rho = 1
        osqp_solver.setup(P=P_sparse, q=c, A=A_sparse, l=l, u=u, alpha=self.alpha, rho=rho, 
                          sigma=self.sigma, polish=False,
                          adaptive_rho=False, scaling=0, max_iter=max_iter, verbose=True, 
                          eps_abs=abs_tol, eps_rel=rel_tol)

        num = z0_mat.shape[0]
        solve_times = np.zeros(num)
        solve_iters = np.zeros(num)
        x_sols = jnp.zeros((num, n))
        y_sols = jnp.zeros((num, m))
        for i in range(num):
            if not self.factor_static_bool:
                P = unvec_symm(q_mat[i, 2 * m + n: 2 * m + n + nc2], n)
                A = jnp.reshape(q_mat[i, 2 * m + n + nc2:], (m, n))
                c, l, u = np.array(q_mat[i, :n]), np.array(q_mat[i, n:n + m]),  np.array(q_mat[i, n + m:n + 2 * m])  # noqa
                
                P_sparse, A_sparse = csc_matrix(np.array(P)), csc_matrix(np.array(A))
                # Px = sparse.triu(P_sparse).data
                # import pdb
                # pdb.set_trace()
                osqp_solver = osqp.OSQP()
                osqp_solver.setup(P=P_sparse, q=c, A=A_sparse, l=l, u=u, alpha=self.alpha, rho=rho, 
                                  sigma=self.sigma, polish=False,
                                  adaptive_rho=False, scaling=0, max_iter=max_iter, verbose=True, 
                                  eps_abs=abs_tol, eps_rel=rel_tol)
                # osqp_solver.update(Px=P_sparse, Ax=csc_matrix(np.array(A)))
            else:
                # set c, l, u
                c, l, u = q_mat[i, :n], q_mat[i, n:n + m], q_mat[i, n + m:n + 2 * m]  # noqa
                osqp_solver.update(q=np.array(c))
                osqp_solver.update(l=np.array(l), u=np.array(u))

            

            # set the warm start
            # x, y, s = self.get_xys_from_z(z0_mat[i, :])
            x_ws, y_ws = np.array(z0_mat[i, :n]), np.array(z0_mat[i, n:n + m])

            # fix warm start
            osqp_solver.warm_start(x=x_ws, y=y_ws)

            # solve
            results = osqp_solver.solve()
            # sol = solver.solve(warm_start=True, x=np.array(x), y=np.array(y), s=np.array(s))

            # set the solve time in seconds
            solve_times[i] = results.info.solve_time * 1000
            solve_iters[i] = results.info.iter

            # set the results
            x_sols = x_sols.at[i, :].set(results.x)
            y_sols = y_sols.at[i, :].set(results.y)

        return solve_times, solve_iters, x_sols, y_sols
