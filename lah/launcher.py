import csv
import os
import time
import gc

import hydra
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import lax
import jax

from jax.interpreters import xla

from functools import partial

from lah.algo_steps import create_projection_fn, get_psd_sizes
from lah.gd_model import GDmodel
from lah.ista_model import ISTAmodel
from lah.lm_gd_model import LMGDmodel
from lah.logisticgd_model import LOGISTICGDmodel
from lah.lm_logisticgd_model import LMLOGISTICGDmodel
from lah.lm_ista_model import LMISTAmodel
from lah.lah_gd_model import LAHGDmodel
from lah.lah_stoch_gd_model import LAHStochasticGDmodel
from lah.lah_logisticgd_model import LAHLOGISTICGDmodel
from lah.lah_ista_model import LAHISTAmodel
from lah.lah_osqp_model import LAHOSQPmodel
from lah.lah_scs_model import LAHSCSmodel
from lah.lm_scs_model import LMSCSmodel
from lah.lm_osqp_model import LMOSQPmodel
from lah.launcher_helper import (
    compute_kl_inv_vector,
    get_nearest_neighbors,
    normalize_inputs_fn,
    plot_samples,
    plot_samples_scs,
    setup_scs_opt_sols,
    geometric_mean,
    stack_tuples
)
from lah.launcher_plotter import (
    plot_eval_iters,
    plot_lah_weights,
    plot_losses_over_examples,
    plot_train_test_losses,
    plot_warm_starts,
    custom_visualize
)
from lah.launcher_writer import (
    create_empty_df,
    test_eval_write,
    update_percentiles,
    write_accuracies_csv,
    write_train_results,
)
from lah.osqp_model import OSQPmodel
from lah.scs_model import SCSmodel
from lah.utils.generic_utils import setup_permutation

# config.update("jax_enable_x64", True)


class Workspace:
    def __init__(self, algo, cfg, static_flag, static_dict, example,
                 traj_length=None,
                 custom_visualize_fn=None,
                 custom_loss=None,
                 shifted_sol_fn=None,
                 closed_loop_rollout_dict=None):
        '''
        cfg is the run_cfg from hydra
        static_flag is True if the matrices P and A don't change from problem to problem
        static_dict holds the data that doesn't change from problem to problem
        example is the string (e.g. 'robust_kalman')
        '''
        self.algo = algo
        if cfg.get('custom_loss', False):
            self.custom_loss = custom_loss
        else:
            self.custom_loss = None
        pac_bayes_cfg = cfg.get('pac_bayes_cfg', {})
        self.skip_pac_bayes_full = pac_bayes_cfg.get('skip_full', True)

        pac_bayes_accs = pac_bayes_cfg.get(
            'frac_solved_accs', 'fp_full')

        if pac_bayes_accs == 'fp_full':
            start = -10  # Start of the log range (log10(10^-5))
            end = 5  # End of the log range (log10(1))
            pac_bayes_accs = list(np.round(np.logspace(start, end, num=151), 10))
        self.frac_solved_accs = pac_bayes_accs
        self.rep = pac_bayes_cfg.get('rep', True)

        self.key_count = 0

        self.static_flag = static_flag
        self.example = example
        self.eval_unrolls = cfg.eval_unrolls + 1
        self.eval_every_x_epochs = cfg.eval_every_x_epochs
        self.save_every_x_epochs = cfg.save_every_x_epochs
        self.num_samples = cfg.get('num_samples', 10)

        self.num_samples_test = cfg.get('num_samples_test', self.num_samples)
        self.num_samples_train = cfg.get('num_samples_train', self.num_samples_test)

        self.eval_batch_size_test = cfg.get('eval_batch_size_test', self.num_samples_test)
        self.eval_batch_size_train = cfg.get('eval_batch_size_train', self.num_samples_train)

        self.key_count = 0
        self.save_weights_flag = cfg.get('save_weights_flag', False)
        self.load_weights_datetime = cfg.get('load_weights_datetime', None)
        self.nn_load_type = cfg.get('nn_load_type', 'deterministic')
        self.shifted_sol_fn = shifted_sol_fn
        self.plot_iterates = cfg.plot_iterates
        self.normalize_inputs = cfg.get('normalize_inputs', True)
        self.epochs_jit = cfg.epochs_jit
        self.accs = cfg.get('accuracies')
        self.no_learning_accs = None
        self.no_learning_pr_dr_max_accs = None
        self.nn_cfg = cfg.nn_cfg

        # custom visualization
        self.init_custom_visualization(cfg, custom_visualize_fn)
        self.vis_num = cfg.get('vis_num', 20)

        # from the run cfg retrieve the following via the data cfg
        N_train, N_test = cfg.N_train, cfg.N_test
        N_val = cfg.get('N_val', 0)
        N = N_train + N_test + N_val
        self.N_val = N_val

        # for control problems only
        self.closed_loop_rollout_dict = closed_loop_rollout_dict
        self.traj_length = traj_length
        if traj_length is not None: # and False:
            self.prev_sol_eval = True
        else:
            self.prev_sol_eval = False

        self.train_unrolls = cfg.train_unrolls

        # load the data from problem to problem
        jnp_load_obj = self.load_setup_data(example, cfg.data.datetime, N_train, N_test, N_val)
        thetas = jnp.array(jnp_load_obj['thetas'])

        self.thetas_train = thetas[self.train_indices, :]
        self.thetas_test = thetas[self.test_indices, :]
        self.thetas_val = thetas[self.val_indices, :]

        train_inputs, test_inputs, normalize_col_sums, normalize_std_dev = normalize_inputs_fn(
            self.normalize_inputs, thetas, self.train_indices, self.test_indices)
        self.train_inputs, self.test_inputs = train_inputs, test_inputs

        

        self.normalize_col_sums, self.normalize_std_dev = normalize_col_sums, normalize_std_dev
        self.skip_startup = cfg.get('skip_startup', False)
        self.setup_opt_sols(algo, jnp_load_obj, N_train, N)

        # progressive train_inputs
        self.lah_train_inputs = 0 * self.z_stars_train
        if algo == 'lah_scs':
            self.lah_train_inputs = jnp.hstack([0 * self.z_stars_train, jnp.ones((N_train, 1))])

        # everything below is specific to the algo
        if algo == 'osqp':
            self.create_osqp_model(cfg, static_dict)
        elif algo == 'scs':
            self.create_scs_model(cfg, static_dict)
        elif algo == 'gd':
            # self.q_mat_train = thetas[:N_train, :]
            # self.q_mat_test = thetas[N_train:N, :]
            self.create_gd_model(cfg, static_dict)
        elif algo == 'ista':
            # self.q_mat_train = thetas[:N_train, :]
            # self.q_mat_test = thetas[N_train:N, :]
            self.create_ista_model(cfg, static_dict)
        elif algo == 'lah_gd':
            self.create_lah_gd_model(cfg, static_dict)
        elif algo == 'lah_stochastic_gd':
            self.create_lah_stochastic_gd_model(cfg, static_dict)
        elif algo == 'lah_logisticgd':
            self.create_lah_logisticgd_model(cfg, static_dict)
        elif algo == 'lm_logisticgd':
            self.create_lm_logisticgd_model(cfg, static_dict)
        elif algo == 'logisticgd':
            self.create_logisticgd_model(cfg, static_dict)
        elif algo == 'lm_gd':
            self.create_lm_gd_model(cfg, static_dict)
        elif algo == 'lah_osqp':
            self.create_lah_osqp_model(cfg, static_dict)
        elif algo == 'lah_scs':
            self.create_lah_scs_model(cfg, static_dict)
        elif algo == 'lm_scs':
            self.create_lm_scs_model(cfg, static_dict)
        elif algo == 'lah_ista':
            self.create_lah_ista_model(cfg, static_dict)
        elif algo == 'lm_ista':
            self.create_lm_ista_model(cfg, static_dict)
        elif algo == 'lm_osqp':
            self.create_lm_osqp_model(cfg, static_dict)
        
    def create_ista_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        A, lambd = static_dict['A'], static_dict['lambd']
        ista_step = static_dict['ista_step']

        input_dict = dict(algorithm='ista',
                        #   supervised=cfg.supervised,
                        #   train_unrolls=self.train_unrolls,
                        #   jit=True,
                        #   train_inputs=self.train_inputs,
                        #   test_inputs=self.test_inputs,
                          b_mat_train=self.q_mat_train,
                          b_mat_test=self.q_mat_test,
                          lambd=lambd,
                          ista_step=ista_step,
                          A=A,
                        #   nn_cfg=cfg.nn_cfg,
                        #   z_stars_train=self.z_stars_train,
                        #   z_stars_test=self.z_stars_test,
                          )
        # self.l2ws_model = ISTAmodel(input_dict)
        self.l2ws_model = ISTAmodel(train_unrolls=self.train_unrolls,
                                    eval_unrolls=self.eval_unrolls,
                                    train_inputs=self.train_inputs,
                                    test_inputs=self.test_inputs,
                                    regression=cfg.supervised,
                                    nn_cfg=cfg.nn_cfg,
                                    z_stars_train=self.z_stars_train,
                                    z_stars_test=self.z_stars_test,
                                    algo_dict=input_dict)


    def create_gd_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        P = static_dict['P']
        gd_step = static_dict['gd_step']

        input_dict = dict(algorithm='gd',
                          c_mat_train=self.q_mat_train,
                          c_mat_test=self.q_mat_test,
                          gd_step=gd_step,
                          P=P
                          )
        self.l2ws_model = GDmodel(train_unrolls=self.train_unrolls,
                                  eval_unrolls=self.eval_unrolls,
                                  train_inputs=self.train_inputs,
                                  test_inputs=self.test_inputs,
                                  regression=cfg.supervised,
                                  nn_cfg=cfg.nn_cfg,
                                  pac_bayes_cfg=cfg.pac_bayes_cfg,
                                  z_stars_train=self.z_stars_train,
                                  z_stars_test=self.z_stars_test,
                                  algo_dict=input_dict)

    def create_lah_gd_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        P = static_dict['P']
        gd_step = static_dict['gd_step']

        input_dict = dict(algorithm='lah_gd',
                          c_mat_train=self.q_mat_train,
                          c_mat_test=self.q_mat_test,
                          gd_step=gd_step,
                          P=P
                          )
        self.l2ws_model = LAHGDmodel(train_unrolls=self.train_unrolls,
                                       eval_unrolls=self.eval_unrolls,
                                       train_inputs=self.train_inputs,
                                       test_inputs=self.test_inputs,
                                       regression=cfg.supervised,
                                       nn_cfg=cfg.nn_cfg,
                                       z_stars_train=self.z_stars_train,
                                       z_stars_test=self.z_stars_test,
                                       loss_method=cfg.loss_method,
                                       algo_dict=input_dict)
        
    def create_lah_stochastic_gd_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        P = static_dict['P']
        gd_step = static_dict['gd_step']
        gauss_mean = static_dict['gauss_mean']
        gauss_var = static_dict['gauss_var']

        input_dict = dict(algorithm='lah_stochastic_gd',
                          c_mat_train=self.q_mat_train,
                          c_mat_test=self.q_mat_test,
                          gd_step=gd_step,
                          P=P,
                          gauss_mean=gauss_mean,
                          gauss_var=gauss_var
                          )
        self.l2ws_model = LAHStochasticGDmodel(train_unrolls=self.train_unrolls,
                                       eval_unrolls=self.eval_unrolls,
                                       train_inputs=self.train_inputs,
                                       test_inputs=self.test_inputs,
                                       regression=cfg.supervised,
                                       nn_cfg=cfg.nn_cfg,
                                       z_stars_train=self.z_stars_train,
                                       z_stars_test=self.z_stars_test,
                                       loss_method=cfg.loss_method,
                                       algo_dict=input_dict)
        
        
    def create_lah_logisticgd_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        num_points = static_dict['num_points']
        gd_step = static_dict['gd_step']

        input_dict = dict(algorithm='lah_logisticgd',
                          q_mat_train=self.q_mat_train,
                          q_mat_test=self.q_mat_test,
                          gd_step=gd_step,
                          num_points=num_points
                          )
        self.l2ws_model = LAHLOGISTICGDmodel(train_unrolls=self.train_unrolls,
                                               step_varying_num=cfg.get('step_varying_num', 50),
                                       eval_unrolls=self.eval_unrolls,
                                       train_inputs=self.train_inputs,
                                       test_inputs=self.test_inputs,
                                       regression=cfg.supervised,
                                       nn_cfg=cfg.nn_cfg,
                                       z_stars_train=self.z_stars_train,
                                       z_stars_test=self.z_stars_test,
                                       loss_method=cfg.loss_method,
                                       algo_dict=input_dict)
        
    def create_lm_logisticgd_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        num_points = static_dict['num_points']
        gd_step = static_dict['gd_step']

        input_dict = dict(algorithm='lm_logisticgd',
                          q_mat_train=self.q_mat_train,
                          q_mat_test=self.q_mat_test,
                          gd_step=gd_step,
                          num_points=num_points
                          )
        self.l2ws_model = LMLOGISTICGDmodel(train_unrolls=self.train_unrolls,
                                       eval_unrolls=self.eval_unrolls,
                                       train_inputs=self.train_inputs,
                                       test_inputs=self.test_inputs,
                                       regression=cfg.supervised,
                                       nn_cfg=cfg.nn_cfg,
                                       z_stars_train=self.z_stars_train,
                                       z_stars_test=self.z_stars_test,
                                       loss_method=cfg.loss_method,
                                       algo_dict=input_dict)
        
    def create_logisticgd_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        num_points = static_dict['num_points']
        gd_step = static_dict['gd_step']

        input_dict = dict(algorithm='lm_logisticgd',
                          q_mat_train=self.q_mat_train,
                          q_mat_test=self.q_mat_test,
                          gd_step=gd_step,
                          num_points=num_points
                          )
        self.l2ws_model = LOGISTICGDmodel(train_unrolls=self.train_unrolls,
                                       eval_unrolls=self.eval_unrolls,
                                       train_inputs=self.train_inputs,
                                       test_inputs=self.test_inputs,
                                       regression=cfg.supervised,
                                       nn_cfg=cfg.nn_cfg,
                                       z_stars_train=self.z_stars_train,
                                       z_stars_test=self.z_stars_test,
                                       loss_method=cfg.loss_method,
                                       algo_dict=input_dict)
        
    def create_lah_ista_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        A = static_dict['A']

        lambd = static_dict['lambd']

        input_dict = dict(algorithm='lah_ista',
                          c_mat_train=self.q_mat_train,
                          c_mat_test=self.q_mat_test,
                          lambd=lambd,
                          A=A
                          )
        self.l2ws_model = LAHISTAmodel(train_unrolls=self.train_unrolls,
                                       eval_unrolls=self.eval_unrolls,
                                       train_inputs=self.train_inputs,
                                       test_inputs=self.test_inputs,
                                       regression=cfg.supervised,
                                       nn_cfg=cfg.nn_cfg,
                                       z_stars_train=self.z_stars_train,
                                       z_stars_test=self.z_stars_test,
                                       loss_method=cfg.loss_method,
                                       algo_dict=input_dict)
        
    def create_lm_ista_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        A = static_dict['A']

        lambd = static_dict['lambd']

        input_dict = dict(algorithm='lm_ista',
                          c_mat_train=self.q_mat_train,
                          c_mat_test=self.q_mat_test,
                          lambd=lambd,
                          A=A
                          )
        self.l2ws_model = LMISTAmodel(train_unrolls=self.train_unrolls,
                                       eval_unrolls=self.eval_unrolls,
                                       train_inputs=self.train_inputs,
                                       test_inputs=self.test_inputs,
                                       regression=cfg.supervised,
                                       nn_cfg=cfg.nn_cfg,
                                       z_stars_train=self.z_stars_train,
                                       z_stars_test=self.z_stars_test,
                                       loss_method=cfg.loss_method,
                                       algo_dict=input_dict)
    
    def create_lm_osqp_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        # A = static_dict['A']

        # lambd = static_dict['lambd']

        # input_dict = dict(algorithm='lm_osqp',
        #                   c_mat_train=self.q_mat_train,
        #                   c_mat_test=self.q_mat_test,
        #                   lambd=lambd,
        #                   A=A
        #                   )
        # self.l2ws_model = LMISTAmodel(train_unrolls=self.train_unrolls,
        #                                eval_unrolls=self.eval_unrolls,
        #                                train_inputs=self.train_inputs,
        #                                test_inputs=self.test_inputs,
        #                                regression=cfg.supervised,
        #                                nn_cfg=cfg.nn_cfg,
        #                                z_stars_train=self.z_stars_train,
        #                                z_stars_test=self.z_stars_test,
        #                                loss_method=cfg.loss_method,
        #                                algo_dict=input_dict)
        

        factor = static_dict['factor']
        A = static_dict['A']
        P = static_dict['P']
        m, n = A.shape
        self.m, self.n = m, n
        rho = static_dict['rho']
        input_dict = dict(algorithm='lm_osqp',
                          factor_static_bool=True,
                          supervised=cfg.supervised,
                          rho=rho,
                          q_mat_train=self.q_mat_train,
                          q_mat_test=self.q_mat_test,
                          A=A,
                          P=P,
                          m=m,
                          n=n,
                          factor=factor,
                          custom_loss=self.custom_loss,
                          plateau_decay=cfg.plateau_decay)
        self.l2ws_model = LMOSQPmodel(train_unrolls=self.train_unrolls,
                                         eval_unrolls=self.eval_unrolls,
                                         train_inputs=self.train_inputs,
                                         test_inputs=self.test_inputs,
                                         regression=cfg.supervised,
                                         nn_cfg=cfg.nn_cfg,
                                         z_stars_train=self.z_stars_train,
                                         z_stars_test=self.z_stars_test,
                                         loss_method=cfg.loss_method,
                                         algo_dict=input_dict)
        
    def create_lm_gd_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        P = static_dict['P']
        gd_step = static_dict['gd_step']

        input_dict = dict(algorithm='lm_gd',
                          c_mat_train=self.q_mat_train,
                          c_mat_test=self.q_mat_test,
                          gd_step=gd_step,
                          P=P
                          )
        self.l2ws_model = LMGDmodel(train_unrolls=self.train_unrolls,
                                       eval_unrolls=self.eval_unrolls,
                                       train_inputs=self.train_inputs,
                                       test_inputs=self.test_inputs,
                                       regression=cfg.supervised,
                                       nn_cfg=cfg.nn_cfg,
                                       z_stars_train=self.z_stars_train,
                                       z_stars_test=self.z_stars_test,
                                       loss_method=cfg.loss_method,
                                       algo_dict=input_dict)

    def create_lah_osqp_model(self, cfg, static_dict):
        factor = static_dict['factor']
        A = static_dict['A']
        P = static_dict['P']
        m, n = A.shape
        self.m, self.n = m, n
        rho = static_dict['rho']
        input_dict = dict(algorithm='lah_osqp',
                          factor_static_bool=True,
                          supervised=cfg.supervised,
                          rho=rho,
                          q_mat_train=self.q_mat_train,
                          q_mat_test=self.q_mat_test,
                          A=A,
                          P=P,
                          num_const_steps=cfg.num_const_steps,
                          m=m,
                          n=n,
                          factor=factor,
                          custom_loss=self.custom_loss,
                          plateau_decay=cfg.plateau_decay)
        self.l2ws_model = LAHOSQPmodel(train_unrolls=self.train_unrolls,
                                         eval_unrolls=self.eval_unrolls,
                                         train_inputs=self.train_inputs,
                                         test_inputs=self.test_inputs,
                                         regression=cfg.supervised,
                                         nn_cfg=cfg.nn_cfg,
                                         z_stars_train=self.z_stars_train,
                                         z_stars_test=self.z_stars_test,
                                         loss_method=cfg.loss_method,
                                         algo_dict=input_dict)

    def create_lah_scs_model(self, cfg, static_dict):
        static_M = static_dict['M']

        # save cones
        self.cones = static_dict['cones_dict']

        self.M = static_M
        proj = create_projection_fn(self.cones, self.n)

        psd_sizes = get_psd_sizes(self.cones)

        self.psd_size = psd_sizes[0]

        algo_dict = {'proj': proj,
                     'q_mat_train': self.q_mat_train,
                     'q_mat_test': self.q_mat_test,
                     'm': self.m,
                     'n': self.n,
                     'static_M': static_M,
                     'num_const_steps': cfg.num_const_steps,
                     'static_flag': self.static_flag,
                     'cones': self.cones,
                     'lightweight': cfg.get('lightweight', False),
                     'custom_loss': self.custom_loss
                     }
        self.l2ws_model = LAHSCSmodel(train_unrolls=self.train_unrolls,
                                        eval_unrolls=self.eval_unrolls,
                                        train_inputs=self.train_inputs,
                                        test_inputs=self.test_inputs,
                                        z_stars_train=self.z_stars_train,
                                        z_stars_test=self.z_stars_test,
                                        x_stars_train=self.x_stars_train,
                                        x_stars_test=self.x_stars_test,
                                        y_stars_train=self.y_stars_train,
                                        y_stars_test=self.y_stars_test,
                                        regression=cfg.get(
                                            'supervised', False),
                                        nn_cfg=cfg.nn_cfg,
                                        loss_method=cfg.loss_method,
                                        algo_dict=algo_dict)
        
    def create_lm_scs_model(self, cfg, static_dict):
        static_M = static_dict['M']

        # save cones
        self.cones = static_dict['cones_dict']

        self.M = static_M
        proj = create_projection_fn(self.cones, self.n)

        psd_sizes = get_psd_sizes(self.cones)

        self.psd_size = psd_sizes[0]

        algo_dict = {'proj': proj,
                     'q_mat_train': self.q_mat_train,
                     'q_mat_test': self.q_mat_test,
                     'm': self.m,
                     'n': self.n,
                     'static_M': static_M,
                     'static_flag': self.static_flag,
                     'cones': self.cones,
                     'lightweight': cfg.get('lightweight', False),
                     'custom_loss': self.custom_loss
                     }
        self.l2ws_model = LMSCSmodel(train_unrolls=self.train_unrolls,
                                        eval_unrolls=self.eval_unrolls,
                                        train_inputs=self.train_inputs,
                                        test_inputs=self.test_inputs,
                                        z_stars_train=self.z_stars_train,
                                        z_stars_test=self.z_stars_test,
                                        x_stars_train=self.x_stars_train,
                                        x_stars_test=self.x_stars_test,
                                        y_stars_train=self.y_stars_train,
                                        y_stars_test=self.y_stars_test,
                                        regression=cfg.get(
                                            'supervised', False),
                                        nn_cfg=cfg.nn_cfg,
                                        loss_method=cfg.loss_method,
                                        algo_dict=algo_dict)

    def create_osqp_model(self, cfg, static_dict):
        factor = static_dict['factor']
        A = static_dict['A']
        P = static_dict['P']
        m, n = A.shape
        self.m, self.n = m, n
        rho = static_dict['rho']
        input_dict = dict(factor_static_bool=True,
                            supervised=cfg.supervised,
                            rho=rho,
                            q_mat_train=self.q_mat_train,
                            q_mat_test=self.q_mat_test,
                            A=A,
                            P=P,
                            m=m,
                            n=n,
                            factor=factor,
                            custom_loss=self.custom_loss,
                            plateau_decay=cfg.plateau_decay)
        
        self.x_stars_train = self.z_stars_train[:, :self.n]
        self.x_stars_test = self.z_stars_test[:, :self.n]
        self.l2ws_model = OSQPmodel(train_unrolls=self.train_unrolls,
                                    eval_unrolls=self.eval_unrolls,
                                    train_inputs=self.train_inputs,
                                    test_inputs=self.test_inputs,
                                    regression=cfg.supervised,
                                    nn_cfg=cfg.nn_cfg,
                                    pac_bayes_cfg=cfg.pac_bayes_cfg,
                                    z_stars_train=self.z_stars_train,
                                    z_stars_test=self.z_stars_test,
                                    algo_dict=input_dict)

    def create_scs_model(self, cfg, static_dict):
        if self.static_flag:
            static_M = static_dict['M']
            static_algo_factor = static_dict['algo_factor']
            cones = static_dict['cones_dict']

        rho_x = cfg.get('rho_x', 1)
        scale = cfg.get('scale', 1)
        alpha_relax = cfg.get('alpha_relax', 1)

        # save cones
        self.cones = static_dict['cones_dict']

        self.M = static_M
        proj = create_projection_fn(cones, self.n)
        psd_sizes = get_psd_sizes(cones)

        self.psd_size = psd_sizes[0]

        algo_dict = {'proj': proj,
                     'q_mat_train': self.q_mat_train,
                     'q_mat_test': self.q_mat_test,
                     'm': self.m,
                     'n': self.n,
                     'static_M': static_M,
                     'static_flag': self.static_flag,
                     'static_algo_factor': static_algo_factor,
                     'rho_x': rho_x,
                     'scale': scale,
                     'alpha_relax': alpha_relax,
                     'cones': cones,
                     'lightweight': cfg.get('lightweight', False),
                     'custom_loss': self.custom_loss
                     }
        self.l2ws_model = SCSmodel(train_unrolls=self.train_unrolls,
                                   eval_unrolls=self.eval_unrolls,
                                   train_inputs=self.train_inputs,
                                   test_inputs=self.test_inputs,
                                   z_stars_train=self.z_stars_train,
                                   z_stars_test=self.z_stars_test,
                                   x_stars_train=self.x_stars_train,
                                   x_stars_test=self.x_stars_test,
                                   y_stars_train=self.y_stars_train,
                                   y_stars_test=self.y_stars_test,
                                   regression=cfg.get('supervised', False),
                                   nn_cfg=cfg.nn_cfg,
                                   pac_bayes_cfg=cfg.pac_bayes_cfg,
                                   algo_dict=algo_dict)

    def setup_opt_sols(self, algo, jnp_load_obj, N_train, N, num_plot=5):
        if algo != 'scs' and algo != 'lah_scs' and algo != 'lm_scs':
            z_stars = jnp_load_obj['z_stars']
            z_stars_train = z_stars[self.train_indices, :]
            z_stars_test = z_stars[self.test_indices, :]
            plot_samples(num_plot, self.thetas_train,
                         self.train_inputs, z_stars_train)
            self.z_stars_test = z_stars_test
            self.z_stars_train = z_stars_train

            # val
            self.z_stars_val = z_stars[self.val_indices, :]
        else:
            opt_train_sols, opt_test_sols, opt_val_sols, self.m, self.n = setup_scs_opt_sols(jnp_load_obj, self.train_indices, self.test_indices, self.val_indices)
            self.x_stars_train, self.y_stars_train, self.z_stars_train = opt_train_sols
            self.x_stars_test, self.y_stars_test, self.z_stars_test = opt_test_sols
            self.x_stars_val, self.y_stars_val, self.z_stars_val = opt_val_sols

            # self.z_stars_val = z_stars[self.val_indices, :]

            plot_samples_scs(num_plot, self.thetas_train, self.train_inputs,
                             self.x_stars_train, self.y_stars_train, self.z_stars_train)


    def save_weights(self):
        if self.l2ws_model.algo[:3] == 'lah':
            self.save_weights_lah()

    def save_weights_lah(self):
        nn_weights = self.l2ws_model.params
        # create directory
        if not os.path.exists('nn_weights'):
            os.mkdir('nn_weights')

        # Save mean weights
        mean_params = nn_weights[0]
        jnp.savez("nn_weights/params.npz", mean_params=mean_params)


    def load_weights(self, example, datetime, nn_type):
        if self.l2ws_model.algo[:3] == 'lah':
            self.load_weights_lah(example, datetime)


    def load_weights_lah(self, example, datetime):
        # get the appropriate folder
        orig_cwd = hydra.utils.get_original_cwd()
        folder = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/nn_weights"

        # load the mean
        loaded_mean = jnp.load(f"{folder}/params.npz")
        mean_params = loaded_mean['mean_params']

        self.l2ws_model.params = [mean_params]


    def normalize_theta(self, theta):
        normalized_input = (theta - self.normalize_col_sums) / \
            self.normalize_std_dev
        return normalized_input

    def load_setup_data(self, example, datetime, N_train, N_test, N_val):
        N = N_train + N_test + N_val
        orig_cwd = hydra.utils.get_original_cwd()
        folder = f"{orig_cwd}/outputs/{example}/data_setup_outputs/{datetime}"
        filename = f"{folder}/data_setup.npz"

        jnp_load_obj = jnp.load(filename)

        if 'q_mat' in jnp_load_obj.keys():
            q_mat = jnp.array(jnp_load_obj['q_mat'])

            # rand_indices = np.random.choice(q_mat.shape[0], N, replace=False)
            rand_indices = np.arange(N)

            # self.train_indices = rand_indices[:N_train]
            self.train_indices = rand_indices[N_test: N_test + N_train]
            self.q_mat_train = q_mat[self.train_indices, :]

            # self.test_indices = rand_indices[N_train:N_train + N_test]
            self.test_indices = rand_indices[:N_test]
            self.q_mat_test = q_mat[self.test_indices, :]

            self.val_indices = rand_indices[N_train + N_test:]
            self.q_mat_val = q_mat[self.val_indices, :]
            
        else:
            thetas = jnp.array(jnp_load_obj['thetas'])
            rand_indices = np.arange(N)
            self.train_indices = rand_indices[N_test: N_test + N_train]
            self.test_indices = rand_indices[:N_test]
            self.val_indices = rand_indices[N_train + N_test:]
            # rand_indices = np.random.choice(thetas.shape[0], N, replace=False)
            # self.train_indices = rand_indices[:N_train]
            # self.test_indices = rand_indices[N_train:]
            # self.val_indices = rand_indices[N_train + N_test:]

            self.q_mat_train = thetas[self.train_indices, :]
            self.q_mat_test = thetas[self.test_indices, :]
            self.q_mat_val = thetas[self.val_indices, :]


        # load the closed_loop_rollout trajectories
        if 'ref_traj_tensor' in jnp_load_obj.keys():
            # load all of the goals
            self.closed_loop_rollout_dict['ref_traj_tensor'] = jnp_load_obj['ref_traj_tensor']

        return jnp_load_obj

    def init_custom_visualization(self, cfg, custom_visualize_fn):
        iterates_visualize = cfg.get('iterates_visualize', 0)
        if custom_visualize_fn is None or iterates_visualize == 0:
            self.has_custom_visualization = False
        else:
            self.has_custom_visualization = True
            self.custom_visualize_fn = custom_visualize_fn
            self.iterates_visualize = iterates_visualize

    def _init_logging(self):
        self.logf = open('log.csv', 'a')
        fieldnames = ['iter', 'train_loss', 'val_loss',
                      'test_loss', 'time_train_per_epoch']
        self.writer = csv.DictWriter(self.logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            self.writer.writeheader()

        self.test_logf = open('log_test.csv', 'a')
        fieldnames = ['iter', 'train_loss', 'test_loss', 'time_per_iter']
        self.test_writer = csv.DictWriter(
            self.test_logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            self.test_writer.writeheader()

        self.logf = open('train_results.csv', 'a')

        fieldnames = ['train_loss', 'moving_avg_train', 'time_train_per_epoch']
        self.writer = csv.DictWriter(self.logf, fieldnames=fieldnames)
        if os.stat('train_results.csv').st_size == 0:
            self.writer.writeheader()

        self.test_logf = open('train_test_results.csv', 'a')
        fieldnames = ['iter', 'train_loss', 'test_loss', 'penalty', 'avg_posterior_var',
                      'stddev_posterior_var', 'prior', 'mean_squared_w', 'time_per_iter']
        self.test_writer = csv.DictWriter(
            self.test_logf, fieldnames=fieldnames)
        if os.stat('train_results.csv').st_size == 0:
            self.test_writer.writeheader()

    def evaluate_iters(self, num, col, train=False, plot=True, plot_pretrain=False):
        if train and col == 'prev_sol':
            return
        fixed_ws = col == 'nesterov' #col == 'nearest_neighbor' or col == 'prev_sol' or col == 'nesterov'
        if col == 'nearest_neighbor' and not self.l2ws_model.lah:
            fixed_ws = True

        # do the actual evaluation (most important step in thie method)
        eval_batch_size = self.eval_batch_size_train if train else self.eval_batch_size_test

        val = col == 'final'
        eval_out = self.evaluate_only(
            fixed_ws, num, train, col, eval_batch_size, val=val)

        # extract information from the evaluation
        loss_train, out_train, train_time = eval_out
        iter_losses_mean = out_train[1].mean(axis=0)

        # plot losses over examples
        losses_over_examples = out_train[1].T

        plot_losses_over_examples(losses_over_examples, train, col)

        # update the eval csv files
        primal_residuals, dual_residuals, obj_vals_diff = None, None, None
        dist_opts = None
        if len(out_train) == 6 or len(out_train) == 8:
            primal_residuals = out_train[4].mean(axis=0)
            dual_residuals = out_train[5].mean(axis=0)
        elif len(out_train) == 5:
            obj_vals_diff = geometric_mean(out_train[4]) #out_train[4].mean(axis=0)
        elif len(out_train) == 9:
            primal_residuals = out_train[4].mean(axis=0)
            dual_residuals = out_train[5].mean(axis=0)
            dist_opts = out_train[8].mean(axis=0)
        if primal_residuals is not None:
            pr_dr_maxes = jnp.maximum(primal_residuals, dual_residuals)
            pr_dr_max = geometric_mean(pr_dr_maxes) #pr_dr_maxes.mean(axis=0)

        if train:
            self.percentiles_df_list_train = update_percentiles(self.percentiles_df_list_train,
                                                                     self.percentiles, 
                                                                     losses_over_examples.T, 
                                                                     train, col)
        else:
            self.percentiles_df_list_test = update_percentiles(self.percentiles_df_list_test,
                                                                     self.percentiles, 
                                                                     losses_over_examples.T, 
                                                                     train, col)

        df_out = self.update_eval_csv(
            iter_losses_mean, train, col,
            primal_residuals=primal_residuals,
            dual_residuals=dual_residuals,
            obj_vals_diff=obj_vals_diff,
            dist_opts=dist_opts
        )
        if primal_residuals is not None:
            pr_dr_max = jnp.maximum(primal_residuals, dual_residuals)
        else:
            pr_dr_max = None
        iters_df, primal_residuals_df, dual_residuals_df, obj_vals_diff_df, dist_opts_df, pr_dr_max_df = df_out

        if not self.skip_startup:
            self.no_learning_accs = write_accuracies_csv(self.accs, iter_losses_mean, train, col, 
                                                         self.no_learning_accs)
            if pr_dr_max is not None:
                self.no_learning_pr_dr_max_accs = write_accuracies_csv(self.accs, pr_dr_max, train, col, 
                                                         self.no_learning_pr_dr_max_accs, pr_dr_max=True)

        # plot the evaluation iterations
        plot_eval_iters(iters_df, primal_residuals_df,
                        dual_residuals_df, plot_pretrain, obj_vals_diff_df, dist_opts_df, pr_dr_max_df, train, 
                        col, self.eval_unrolls, self.train_unrolls)

        # plot the warm-start predictions
        z_all = out_train[2]
        

        if isinstance(self.l2ws_model, SCSmodel) or isinstance(self.l2ws_model, LAHSCSmodel) or isinstance(self.l2ws_model, LMSCSmodel):
            # out_train[6]
            z_plot = z_all[:, :, :-1] / (z_all[:, :, -1:] + 1e-10)

            u_all = out_train[6]

            u_plot = u_all[:, :, :self.l2ws_model.n] / (u_all[:, :, -1:]) # + 1e-10)
            z_plot = u_plot
        else:
            z_plot = z_all

        plot_warm_starts(self.l2ws_model, self.plot_iterates, z_plot, train, col)

        if self.l2ws_model.algo[:3] == 'lah':
            n_iters = 64 if col == 'silver' else self.l2ws_model.step_varying_num + 1 #51
            if col == 'silver' or col == 'conj_grad':
                transformed_params = self.l2ws_model.params[0]
            else:
                transformed_params = self.l2ws_model.transform_params(self.l2ws_model.params, n_iters)

            plot_lah_weights(transformed_params, col)

        # custom visualize
        z_stars = self.z_stars_train if train else self.z_stars_test
        thetas = self.thetas_train if train else self.thetas_test
        if not hasattr(self, 'z_no_learn_train') and train:
            self.z_no_learn_train = z_plot
        elif not hasattr(self, 'z_no_learn_test') and not train:
            self.z_no_learn_test = z_plot
        z_no_learn = self.z_no_learn_train if train else self.z_no_learn_test
        z_nn = z_no_learn
        z_prev_sol = z_no_learn
        if self.has_custom_visualization:
            if self.vis_num > 0:
                # custom_visualize(z_plot, train, col)
                # u_plot = z_plot
                custom_visualize(self.custom_visualize_fn, self.iterates_visualize, self.vis_num, 
                                 thetas, z_plot, z_stars, z_no_learn, z_nn, z_prev_sol, train, col)

        if self.save_weights_flag:
            self.save_weights()
        gc.collect()


        return out_train

    def run(self):
        # setup logging and dataframes
        self._init_logging()
        self.setup_dataframes()

        if not self.skip_startup:
            # no learning evaluation
            self.eval_iters_train_and_test('no_train', None)

            # if self.l2ws_model.lah:
            # nearest neighbor
            if self.l2ws_model.lah:
                self.eval_iters_train_and_test('nearest_neighbor', None)
                jax.clear_caches()

            if self.l2ws_model.algo == 'lah_ista':
                # nesterov
                self.l2ws_model.set_params_for_nesterov()
                self.eval_iters_train_and_test('nesterov', None)

            if self.l2ws_model.algo == 'lah_gd' or self.l2ws_model.algo == 'lah_stochastic_gd':
                # conj_grad
                self.eval_iters_train_and_test('conj_grad', None)

                # nesterov
                self.l2ws_model.set_params_for_nesterov()
                self.eval_iters_train_and_test('nesterov', None)

                # silver
                self.l2ws_model.set_params_for_silver()
                self.eval_iters_train_and_test('silver', None)

                # perturb slightly for training
                self.l2ws_model.perturb_params()

            elif self.l2ws_model.algo == 'lah_logisticgd':
                # nesterov
                self.l2ws_model.set_params_for_nesterov()
                self.eval_iters_train_and_test('nesterov', None)

                # silver
                self.l2ws_model.set_params_for_silver()
                self.eval_iters_train_and_test('silver', None)

                # perturb slightly for training
                self.l2ws_model.init_params()

            # prev sol eval
            if 'lah' in self.l2ws_model.algo and self.prev_sol_eval and self.l2ws_model.z_stars_train is not None:
                self.eval_iters_train_and_test('prev_sol', None)

        # load the weights AFTER the cold-start
        if self.load_weights_datetime is not None:
            self.load_weights(
                self.example, self.load_weights_datetime, self.nn_load_type)

        # eval test data to start
        self.test_writer, self.test_logf, self.l2ws_model = test_eval_write(self.test_writer, 
                                                                            self.test_logf, 
                                                                            self.l2ws_model)

        # do all of the training
        test_zero = True if self.skip_startup else False
        self.train(test_zero=test_zero)

    def train(self, test_zero=False):
        """
        does all of the training
        jits together self.epochs_jit number of epochs together
        writes results to filesystem
        """
        num_epochs_jit = int(self.l2ws_model.epochs / self.epochs_jit)
        loop_size = int(self.l2ws_model.num_batches * self.epochs_jit)

        num_progressive_trains = int(self.l2ws_model.step_varying_num * self.l2ws_model.num_const_steps / self.train_unrolls + 1)

        

        for window in range(num_progressive_trains):
            # update window_indices
            window_indices = jnp.arange(int(self.train_unrolls * window / self.l2ws_model.num_const_steps), 
                                        int(self.train_unrolls * (window + 1) / self.l2ws_model.num_const_steps))
            steady_state = window == num_progressive_trains - 1

            # update the train inputs
            update_train_inputs_flag = window > 0 and self.l2ws_model.lah

            # update the method for the steady-state
            # if window == num_progressive_trains - 1:
            #     self.l2ws_model.train_case = 'gradient'

            for epoch_batch in range(num_epochs_jit):
                epoch = int(epoch_batch * self.epochs_jit) + window * num_epochs_jit * self.epochs_jit
                print('epoch', epoch)

                if (test_zero and epoch == 0) or (epoch % self.eval_every_x_epochs == 0 and epoch > 0):
                    if update_train_inputs_flag:
                        self.eval_iters_train_and_test(f"train_epoch_{epoch}",self.train_unrolls * window)
                        update_train_inputs_flag = False
                        jax.clear_caches()
                    else:
                        self.eval_iters_train_and_test(f"train_epoch_{epoch}", None)

                # setup the permutations
                permutation = setup_permutation(
                    self.key_count, self.l2ws_model.N_train, self.epochs_jit)
                
                # import pdb
                # pdb.set_trace()
                prev_params = self.l2ws_model.params[0]

                # train the jitted epochs
                curr_params, state, epoch_train_losses, time_train_per_epoch = self.train_jitted_epochs(
                    permutation, epoch, window_indices, steady_state=steady_state)
                # import pdb
                # pdb.set_trace()
                
                self.l2ws_model.params = [prev_params]
                # import pdb
                # pdb.set_trace()
                
                # insert the curr_params into the entire params
                if self.l2ws_model.lah:
                    # import pdb
                    # pdb.set_trace()
                    pp = self.l2ws_model.params[0].at[window_indices, :].set(curr_params[0])
                    params = [pp]
                else:
                    params = curr_params

                # reset the global (params, state)
                self.key_count += 1
                self.l2ws_model.epoch += self.epochs_jit
                self.l2ws_model.params, self.l2ws_model.state = params, state

                gc.collect()

                prev_batches = len(self.l2ws_model.tr_losses_batch)
                self.l2ws_model.tr_losses_batch = self.l2ws_model.tr_losses_batch + \
                    list(epoch_train_losses)

                # write train results
                self.writer, self.logf = write_train_results(self.writer, self.logf, 
                                                                self.l2ws_model.tr_losses_batch, 
                                                                loop_size, prev_batches,
                                                                epoch_train_losses, 
                                                                time_train_per_epoch)

                # evaluate the test set and write results
                self.test_writer, self.test_logf, self.l2ws_model = test_eval_write(self.test_writer, 
                                                                                self.test_logf, 
                                                                                self.l2ws_model)

                # plot the train / test loss so far
                if epoch % self.save_every_x_epochs == 0:
                    plot_train_test_losses(self.l2ws_model.tr_losses_batch,
                                        self.l2ws_model.te_losses,
                                        self.l2ws_model.num_batches, self.epochs_jit)
        self.eval_iters_train_and_test(f"train_epoch_{epoch}_final", None)
        self.get_confidence_bands()            
        

    def get_confidence_bands(self):
        # do the final evaluation
        out_train = self.evaluate_iters(self.num_samples_test, 'final', train=False)

        if len(out_train) == 6 or len(out_train) == 8:
            primal_residuals = out_train[4] #.mean(axis=0)
            dual_residuals = out_train[5] #.mean(axis=0)
            pr_dr_maxes = jnp.maximum(primal_residuals, dual_residuals)
        elif len(out_train) == 5:
            obj_diffs = out_train[4] #.mean(axis=0)
        elif len(out_train) == 9:
            primal_residuals = out_train[4] #.mean(axis=0)
            dual_residuals = out_train[5] #.mean(axis=0)
            dist_opts = out_train[8].mean(axis=0)
            if primal_residuals is not None:
                pr_dr_maxes = jnp.maximum(primal_residuals, dual_residuals)

        # loop over metric
        if len(out_train) in [9, 6, 8]:
            metric_names = ['primal_residuals', 'dual_residuals', 'pr_dr_maxes']
            metrics = [primal_residuals, dual_residuals, pr_dr_maxes]
        elif len(out_train) == 5:
            metric_names = ['obj_diffs']
            metrics = [obj_diffs]
        
        for j in range(len(metrics)):
            metric = metrics[j]
            metric_name = metric_names[j]
            for i in range(len(self.frac_solved_accs)):
                # compute frac solved
                fs = (metric < self.frac_solved_accs[i])
                emp_success_rates = fs.mean(axis=0) # a vector over the iterations
                emp_risks = 1 - emp_success_rates  # a vector over the iterations

                upper_risk_bounds = compute_kl_inv_vector(emp_risks, self.l2ws_model.delta, 
                                                        self.N_val)
                lower_risk_bounds = 1 - compute_kl_inv_vector(emp_success_rates, self.l2ws_model.delta, 
                                                            self.N_val)
                if not os.path.exists(f"frac_solved_{metric_name}"):
                    os.mkdir(f"frac_solved_{metric_name}")
                filename = f"frac_solved_{metric_name}/tol={self.frac_solved_accs[i]}"
                curr_df = self.frac_solved_df_list_test[i]
                curr_df['empirical_risk'] = emp_risks
                curr_df['upper_risk_bound'] = upper_risk_bounds
                curr_df['lower_risk_bound'] = lower_risk_bounds

                # plot and update csv
                csv_filename = filename + '_test.csv'
                curr_df.to_csv(csv_filename)


    def train_jitted_epochs(self, permutation, epoch, window_indices, steady_state):
        """
        train self.epochs_jit at a time
        special case: the first time we call train_batch (i.e. epoch = 0)
        """
        epoch_batch_start_time = time.time()
        loop_size = int(self.l2ws_model.num_batches * self.epochs_jit)
        epoch_train_losses = jnp.zeros(loop_size)
        n_iters = 1 if steady_state else int(self.l2ws_model.train_unrolls)
        print('n_iters', n_iters)

        if epoch == 0:
            # unroll the first iterate so that This allows `init_val` and `body_fun`
            #   below to have the same output type, which is a requirement of
            #   lax.while_loop and lax.scan.
            batch_indices = lax.dynamic_slice(
                permutation, (0,), (self.l2ws_model.batch_size,))

            if self.l2ws_model.lah:
                if 'stochastic' in self.l2ws_model.algo:
                    
                    train_loss_first, params, state = self.l2ws_model.train_batch(
                        batch_indices, [self.l2ws_model.params[0][:window_indices[0]]], [self.l2ws_model.params[0][window_indices, :]], 
                        self.l2ws_model.state, n_iters=n_iters, train_case=self.l2ws_model.train_case)
                else:
                    train_loss_first, params, state = self.l2ws_model.train_batch(
                        batch_indices, self.l2ws_model.lah_train_inputs, [self.l2ws_model.params[0][window_indices, :]], 
                        self.l2ws_model.state, n_iters=n_iters, train_case=self.l2ws_model.train_case)
            else:
                train_loss_first, params, state = self.l2ws_model.train_batch(
                    batch_indices, self.l2ws_model.train_inputs, self.l2ws_model.params, 
                    self.l2ws_model.state, n_iters=n_iters)
            
            epoch_train_losses = epoch_train_losses.at[0].set(train_loss_first)
            start_index = 1
        else:
            start_index = 0
            if self.l2ws_model.lah:
                params, state = [self.l2ws_model.params[0][window_indices, :]], self.l2ws_model.state
            else:
                params, state = self.l2ws_model.params, self.l2ws_model.state
        
        train_over_epochs_body_simple_fn_jitted = partial(self.train_over_epochs_body_simple_fn, 
                                                          n_iters=n_iters)

        if self.l2ws_model.lah:
            prev_params = self.l2ws_model.params[0]
            if 'stochastic' in self.l2ws_model.algo:
                init_val = epoch_train_losses, [self.l2ws_model.params[0][:window_indices[0]]], params, state, permutation
            else:
                init_val = epoch_train_losses, self.l2ws_model.lah_train_inputs, params, state, permutation
        else:
            init_val = epoch_train_losses, self.l2ws_model.train_inputs, params, state, permutation
        val = lax.fori_loop(start_index, loop_size,
                            train_over_epochs_body_simple_fn_jitted, init_val)
        epoch_batch_end_time = time.time()
        time_diff = epoch_batch_end_time - epoch_batch_start_time
        time_train_per_epoch = time_diff / self.epochs_jit
        epoch_train_losses, inputs, params, state, permutation = val
        print('epoch_train_losses', epoch_train_losses)

        self.l2ws_model.key = state.iter_num

        return params, state, epoch_train_losses, time_train_per_epoch

    def train_over_epochs_body_simple_fn(self, batch, val, n_iters):
        """
        to be used as the body_fn in lax.fori_loop
        need to call partial for the specific permutation
        """
        train_losses, inputs, params, state, permutation = val
        start_index = batch * self.l2ws_model.batch_size
        batch_indices = lax.dynamic_slice(
            permutation, (start_index,), (self.l2ws_model.batch_size,))

        train_loss, params, state = self.l2ws_model.train_batch(
            batch_indices, inputs, params, state, n_iters, train_case=self.l2ws_model.train_case)

        train_losses = train_losses.at[batch].set(train_loss)
        val = train_losses, inputs, params, state, permutation
        return val


    def eval_iters_train_and_test(self, col, new_start_index):
        self.evaluate_iters(
            self.num_samples_test, col, train=False)
        out_train = self.evaluate_iters(
            self.num_samples_train, col, train=True)
        
        # update self.l2ws_model.train_inputs
        print('new_start_index', new_start_index)
        if new_start_index is not None:
            # k = self.train_unrolls
            # self.evaluate_diff_only(k, self.l2ws_model.train_inputs, [self.l2ws_model.params[0][:1, :]])
            if self.algo == 'lah_scs':
                self.l2ws_model.lah_train_inputs = out_train[2][:, new_start_index, :] #-1]
                self.l2ws_model.k_steps_train_fn = self.l2ws_model.k_steps_train_fn2
            elif self.algo == 'lah_osqp':
                m, n = self.l2ws_model.m, self.l2ws_model.n
                self.l2ws_model.lah_train_inputs = out_train[2][:, new_start_index, :m + n]
            else:
                self.l2ws_model.lah_train_inputs = out_train[2][:, new_start_index, :]
            self.l2ws_model.reinit_losses()
            self.l2ws_model.init_optimizer()
                

    def evaluate_diff_only(self, k, inputs, params):
        if inputs is None:
            inputs = self.l2ws_model.train_inputs
        # self.params, inputs, b, k, z_stars, key, factors
        return self.l2ws_model.loss_fn_train(params,
                                             inputs, 
                                             self.l2ws_model.q_mat_train,
                                             k, 
                                             self.l2ws_model.z_stars_train,
                                             0)

    def evaluate_only(self, fixed_ws, num, train, col, batch_size, val=False):
        tag = 'train' if train else 'test'
        factors = None

        if train:
            z_stars = self.l2ws_model.z_stars_train[:num, :]
        else:
            if val:
                z_stars = self.z_stars_val
            else:
                z_stars = self.l2ws_model.z_stars_test[:num, :]
        if col == 'prev_sol':
            if train:
                q_mat_full = self.l2ws_model.q_mat_train[:num, :]
            else:
                q_mat_full = self.l2ws_model.q_mat_test[:num, :]
            non_first_indices = jnp.mod(jnp.arange(num), self.traj_length) != 0
            q_mat = q_mat_full[non_first_indices, :]
            z_stars = z_stars[non_first_indices, :]
        else:
            q_mat = self.l2ws_model.q_mat_train[:num,
                                                :] if train else self.l2ws_model.q_mat_test[:num, :]
            if val:
                q_mat = self.q_mat_val

        z0_inits = self.get_inputs_for_eval(fixed_ws, num, train, col)

        # do the batching
        num_batches = int(num / batch_size)
        full_eval_out = []
        key = 64 if col == 'silver' else 1 + self.l2ws_model.step_varying_num
        if num_batches <= 1:
            

            eval_out = self.l2ws_model.evaluate(
                self.eval_unrolls, z0_inits, q_mat, z_stars, fixed_ws, key, factors=factors, tag=col)
            return eval_out
        for i in range(num_batches):
            print('evaluating batch num', i)
            start = i * batch_size
            end = (i + 1) * batch_size
            curr_z0_inits = z0_inits[start: end]
            curr_q_mat = q_mat[start: end]

            if factors is not None:
                curr_factors = (factors[0][start:end, :, :], factors[1][start:end, :])
            else:
                curr_factors = None
            if z_stars is not None:
                curr_z_stars = z_stars[start: end]
            else:
                curr_z_stars = None
            eval_out = self.l2ws_model.evaluate(
                self.eval_unrolls, curr_z0_inits, curr_q_mat, curr_z_stars, fixed_ws, key,
                factors=curr_factors, tag=tag)
            # full_eval_out.append(eval_out)
            # eval_out_cpu = tuple(item.copy_to_host() for item in eval_out)
            # full_eval_out.append(eval_out_cpu)
            eval_out1_list = [eval_out[1][i] for i in range(len(eval_out[1]))]
            eval_out1_list[2] = eval_out1_list[2][:10, :22, :]
            if isinstance(self.l2ws_model, SCSmodel) or isinstance(self.l2ws_model, LAHSCSmodel):
                eval_out1_list[6] = eval_out1_list[6][:10, :22, :]
                eval_out1_list[7] = eval_out1_list[7][:10, :22, :]
            eval_out_cpu = (eval_out[0], tuple(eval_out1_list), eval_out[2])
            full_eval_out.append(eval_out_cpu)
            del eval_out
            del eval_out_cpu
            del eval_out1_list
            gc.collect()
        loss = np.array([curr_out[0] for curr_out in full_eval_out]).mean()
        time_per_prob = np.array([curr_out[2] for curr_out in full_eval_out]).mean()
        out = stack_tuples([curr_out[1] for curr_out in full_eval_out])

        flattened_eval_out = (loss, out, time_per_prob)

        return flattened_eval_out


    def get_inputs_for_eval(self, fixed_ws, num, train, col):
        if col == 'nearest_neighbor':
            is_osqp = isinstance(self.l2ws_model, OSQPmodel) or isinstance(self.l2ws_model, LAHOSQPmodel)
            if is_osqp:
                m, n = self.l2ws_model.m, self.l2ws_model.n
            else:
                m, n = 0, 0
            inputs = get_nearest_neighbors(is_osqp, 
                                           self.thetas_train,
                                           self.thetas_test,
                                            self.l2ws_model.z_stars_train,
                                            train, num, m=m, n=n)

        elif col == 'prev_sol':
            # now set the indices (0, num_traj, 2 * num_traj) to zero
            non_last_indices = jnp.mod(jnp.arange(
                num), self.traj_length) != self.traj_length - 1
            inputs = self.shifted_sol_fn(
                self.z_stars_test[:num, :][non_last_indices, :])
        else:
            if self.l2ws_model.lah:
                if isinstance(self.l2ws_model, LAHOSQPmodel):
                    m, n = self.l2ws_model.m, self.l2ws_model.n
                    if train:
                        inputs = self.l2ws_model.z_stars_train[:num, :m + n] * 0
                    else:
                        inputs = self.l2ws_model.z_stars_test[:num, :m + n] * 0
                else:
                    if train:
                        inputs = self.l2ws_model.z_stars_train[:num, :] * 0
                    else:
                        inputs = self.l2ws_model.z_stars_test[:num, :] * 0
            else:
                if train:
                    inputs = self.l2ws_model.train_inputs[:num, :]
                else:
                    inputs = self.l2ws_model.test_inputs[:num, :]
        if self.l2ws_model.algo == 'lah_scs':
            inputs = jnp.hstack([inputs, jnp.ones((inputs.shape[0], 1))])
        return inputs


    def setup_dataframes(self):
        self.iters_df_train = create_empty_df(self.eval_unrolls)
        self.iters_df_test = create_empty_df(self.eval_unrolls)

        # primal and dual residuals
        self.primal_residuals_df_train = create_empty_df(self.eval_unrolls)
        self.dual_residuals_df_train = create_empty_df(self.eval_unrolls)
        self.primal_residuals_df_test = create_empty_df(self.eval_unrolls)
        self.dual_residuals_df_test = create_empty_df(self.eval_unrolls)
        self.pr_dr_max_df_train = create_empty_df(self.eval_unrolls)
        self.pr_dr_max_df_test = create_empty_df(self.eval_unrolls)

        # obj_vals_diff
        self.obj_vals_diff_df_train = create_empty_df(self.eval_unrolls)
        self.obj_vals_diff_df_test = create_empty_df(self.eval_unrolls)

        # dist_opts
        self.dist_opts_df_train = create_empty_df(self.eval_unrolls)
        self.dist_opts_df_test = create_empty_df(self.eval_unrolls)

        self.frac_solved_df_list_train, self.frac_solved_df_list_test = [], []
        for i in range(len(self.frac_solved_accs)):
            self.frac_solved_df_list_train.append(pd.DataFrame(columns=['iterations']))
            self.frac_solved_df_list_test.append(pd.DataFrame(columns=['iterations']))
        if not os.path.exists('frac_solved'):
            os.mkdir('frac_solved')

        self.percentiles = [10, 20, 30, 40, 50, 60, 70, 80,
                            90, 95, 96, 97, 98, 99]
        self.percentiles_df_list_train, self.percentiles_df_list_test = [], []
        for i in range(len(self.percentiles)):
            self.percentiles_df_list_train.append(pd.DataFrame(columns=['iterations']))
            self.percentiles_df_list_test.append(pd.DataFrame(columns=['iterations']))


    def update_eval_csv(self, iter_losses_mean, train, col, primal_residuals=None,
                        dual_residuals=None, obj_vals_diff=None, dist_opts=None):
        """
        update the eval csv files
            fixed point residuals
            primal residuals
            dual residuals
        returns the new dataframes
        """
        primal_residuals_df, dual_residuals_df, pr_dr_df = None, None, None
        obj_vals_diff_df = None
        dist_opts_df = None
        if train:
            self.iters_df_train[col] = iter_losses_mean
            self.iters_df_train.to_csv('iters_compared_train.csv')
            if primal_residuals is not None:
                self.primal_residuals_df_train[col] = primal_residuals
                self.primal_residuals_df_train.to_csv('primal_residuals_train.csv')
                self.dual_residuals_df_train[col] = dual_residuals
                self.dual_residuals_df_train.to_csv('dual_residuals_train.csv')
                self.pr_dr_max_df_train[col] = jnp.maximum(primal_residuals, dual_residuals)
                self.pr_dr_max_df_train.to_csv('pr_dr_max_train.csv')

                primal_residuals_df = self.primal_residuals_df_train
                dual_residuals_df = self.dual_residuals_df_train
                pr_dr_df = self.pr_dr_max_df_train
            if obj_vals_diff is not None:
                self.obj_vals_diff_df_train[col] = obj_vals_diff
                self.obj_vals_diff_df_train.to_csv('obj_vals_diff_train.csv')
                obj_vals_diff_df = self.obj_vals_diff_df_train
            if dist_opts is not None:
                self.dist_opts_df_train[col] = dist_opts
                self.dist_opts_df_train.to_csv('dist_opts_df_train.csv')
                dist_opts_df = self.dist_opts_df_train
            iters_df = self.iters_df_train

        else:
            self.iters_df_test[col] = iter_losses_mean
            self.iters_df_test.to_csv('iters_compared_test.csv')
            if primal_residuals is not None:
                self.primal_residuals_df_test[col] = primal_residuals
                self.primal_residuals_df_test.to_csv('primal_residuals_test.csv')
                self.dual_residuals_df_test[col] = dual_residuals
                self.dual_residuals_df_test.to_csv('dual_residuals_test.csv')
                self.pr_dr_max_df_test[col] = jnp.maximum(primal_residuals, dual_residuals)
                self.pr_dr_max_df_test.to_csv('pr_dr_max_test.csv')

                primal_residuals_df = self.primal_residuals_df_test
                dual_residuals_df = self.dual_residuals_df_test
                pr_dr_df = self.pr_dr_max_df_test
            if obj_vals_diff is not None:
                self.obj_vals_diff_df_test[col] = obj_vals_diff
                self.obj_vals_diff_df_test.to_csv('obj_vals_diff_test.csv')
                obj_vals_diff_df = self.obj_vals_diff_df_test
            if dist_opts is not None:
                self.dist_opts_df_test[col] = dist_opts
                self.dist_opts_df_test.to_csv('dist_opts_df_test.csv')
                dist_opts_df = self.dist_opts_df_test

            iters_df = self.iters_df_test

        return iters_df, primal_residuals_df, dual_residuals_df, obj_vals_diff_df, dist_opts_df, pr_dr_df
