import csv
import os
import time

import hydra
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import lax

from functools import partial

from lasco.algo_steps import create_projection_fn, get_psd_sizes
from lasco.gd_model import GDmodel
from lasco.lasco_gd_model import LASCOGDmodel
from lasco.lasco_osqp_model import LASCOOSQPmodel
from lasco.lasco_scs_model import LASCOSCSmodel
from lasco.launcher_helper import (
    get_nearest_neighbors,
    normalize_inputs_fn,
    plot_samples,
    plot_samples_scs,
    setup_scs_opt_sols,
)
from lasco.launcher_plotter import (
    plot_eval_iters,
    plot_lasco_weights,
    plot_losses_over_examples,
    plot_train_test_losses,
    plot_warm_starts,
)
from lasco.launcher_writer import (
    create_empty_df,
    test_eval_write,
    update_percentiles,
    write_accuracies_csv,
    write_train_results,
)
from lasco.osqp_model import OSQPmodel
from lasco.scs_model import SCSmodel
from lasco.utils.generic_utils import setup_permutation

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
            'frac_solved_accs', [0.1, 0.01, 0.001, 0.0001])

        if pac_bayes_accs == 'fp_full':
            start = -6  # Start of the log range (log10(10^-5))
            end = 2  # End of the log range (log10(1))
            pac_bayes_accs = list(np.round(np.logspace(start, end, num=81), 6))
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
        self.nn_cfg = cfg.nn_cfg

        # custom visualization
        self.init_custom_visualization(cfg, custom_visualize_fn)
        self.vis_num = cfg.get('vis_num', 20)

        # from the run cfg retrieve the following via the data cfg
        N_train, N_test = cfg.N_train, cfg.N_test
        N = N_train + N_test

        # for control problems only
        self.closed_loop_rollout_dict = closed_loop_rollout_dict
        self.traj_length = traj_length
        if traj_length is not None and False:
            self.prev_sol_eval = True
        else:
            self.prev_sol_eval = False

        self.train_unrolls = cfg.train_unrolls

        # load the data from problem to problem
        jnp_load_obj = self.load_setup_data(example, cfg.data.datetime, N_train, N)
        thetas = jnp.array(jnp_load_obj['thetas'])
        self.thetas_train = thetas[:N_train, :]
        self.thetas_test = thetas[N_train:N, :]

        train_inputs, test_inputs, normalize_col_sums, normalize_std_dev = normalize_inputs_fn(
            self.normalize_inputs, thetas, N_train, N_test)
        self.train_inputs, self.test_inputs = train_inputs, test_inputs

        

        self.normalize_col_sums, self.normalize_std_dev = normalize_col_sums, normalize_std_dev
        self.skip_startup = cfg.get('skip_startup', False)
        self.setup_opt_sols(algo, jnp_load_obj, N_train, N)

        # progressive train_inputs
        self.train_inputs = 0 * self.z_stars_train

        # everything below is specific to the algo
        if algo == 'osqp':
            self.create_osqp_model(cfg, static_dict)
        elif algo == 'scs':
            self.create_scs_model(cfg, static_dict)
        elif algo == 'gd':
            self.q_mat_train = thetas[:N_train, :]
            self.q_mat_test = thetas[N_train:N, :]
            self.create_gd_model(cfg, static_dict)
        elif algo == 'lasco_gd':
            self.q_mat_train = thetas[:N_train, :]
            self.q_mat_test = thetas[N_train:N, :]
            self.create_lasco_gd_model(cfg, static_dict)
        elif algo == 'lasco_osqp':
            self.create_lasco_osqp_model(cfg, static_dict)
        elif algo == 'lasco_scs':
            self.create_lasco_scs_model(cfg, static_dict)
        
        


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

    def create_lasco_gd_model(self, cfg, static_dict):
        # get A, lambd, ista_step
        P = static_dict['P']
        gd_step = static_dict['gd_step']

        input_dict = dict(algorithm='lasco_gd',
                          c_mat_train=self.q_mat_train,
                          c_mat_test=self.q_mat_test,
                          gd_step=gd_step,
                          P=P
                          )
        self.l2ws_model = LASCOGDmodel(train_unrolls=self.train_unrolls,
                                       eval_unrolls=self.eval_unrolls,
                                       train_inputs=self.train_inputs,
                                       test_inputs=self.test_inputs,
                                       regression=cfg.supervised,
                                       nn_cfg=cfg.nn_cfg,
                                       pac_bayes_cfg=cfg.pac_bayes_cfg,
                                       z_stars_train=self.z_stars_train,
                                       z_stars_test=self.z_stars_test,
                                       loss_method=cfg.loss_method,
                                       algo_dict=input_dict)

    def create_lasco_osqp_model(self, cfg, static_dict):
        factor = static_dict['factor']
        A = static_dict['A']
        P = static_dict['P']
        m, n = A.shape
        self.m, self.n = m, n
        rho = static_dict['rho']
        input_dict = dict(algorithm='lasco_osqp',
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
        self.l2ws_model = LASCOOSQPmodel(train_unrolls=self.train_unrolls,
                                         eval_unrolls=self.eval_unrolls,
                                         train_inputs=self.train_inputs,
                                         test_inputs=self.test_inputs,
                                         regression=cfg.supervised,
                                         nn_cfg=cfg.nn_cfg,
                                         pac_bayes_cfg=cfg.pac_bayes_cfg,
                                         z_stars_train=self.z_stars_train,
                                         z_stars_test=self.z_stars_test,
                                         loss_method=cfg.loss_method,
                                         algo_dict=input_dict)

    def create_lasco_scs_model(self, cfg, static_dict):
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
        self.l2ws_model = LASCOSCSmodel(train_unrolls=self.train_unrolls,
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
                                        pac_bayes_cfg=cfg.pac_bayes_cfg,
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
        if algo != 'scs' and algo != 'lasco_scs':
            z_stars = jnp_load_obj['z_stars']
            z_stars_train = z_stars[:N_train, :]
            z_stars_test = z_stars[N_train:N, :]
            plot_samples(num_plot, self.thetas_train,
                         self.train_inputs, z_stars_train)
            self.z_stars_test = z_stars_test
            self.z_stars_train = z_stars_train
        else:
            opt_train_sols, opt_test_sols, self.m, self.n = setup_scs_opt_sols(jnp_load_obj, N_train, N)
            self.x_stars_train, self.y_stars_train, self.z_stars_train = opt_train_sols
            self.x_stars_test, self.y_stars_test, self.z_stars_test = opt_test_sols

            plot_samples_scs(num_plot, self.thetas_train, self.train_inputs,
                             self.x_stars_train, self.y_stars_train, self.z_stars_train)


    def save_weights(self):
        if self.l2ws_model.algo[:5] == 'lasco':
            self.save_weights_lasco()

    def save_weights_lasco(self):
        nn_weights = self.l2ws_model.params
        # create directory
        if not os.path.exists('nn_weights'):
            os.mkdir('nn_weights')

        # Save mean weights
        mean_params = nn_weights[0]
        jnp.savez("nn_weights/params.npz", mean_params=mean_params)


    def load_weights(self, example, datetime, nn_type):
        if self.l2ws_model.algo[:5] == 'lasco':
            self.load_weights_lasco(example, datetime)


    def load_weights_lasco(self, example, datetime):
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

    def load_setup_data(self, example, datetime, N_train, N):
        orig_cwd = hydra.utils.get_original_cwd()
        folder = f"{orig_cwd}/outputs/{example}/data_setup_outputs/{datetime}"
        filename = f"{folder}/data_setup.npz"

        jnp_load_obj = jnp.load(filename)

        if 'q_mat' in jnp_load_obj.keys():
            q_mat = jnp.array(jnp_load_obj['q_mat'])
            q_mat_train = q_mat[:N_train, :]
            q_mat_test = q_mat[N_train:N, :]
            self.q_mat_train, self.q_mat_test = q_mat_train, q_mat_test

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
        fixed_ws = col == 'nearest_neighbor' or col == 'prev_sol'

        # do the actual evaluation (most important step in thie method)
        eval_batch_size = self.eval_batch_size_train if train else self.eval_batch_size_test
        eval_out = self.evaluate_only(
            fixed_ws, num, train, col, eval_batch_size)

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
            obj_vals_diff = out_train[4].mean(axis=0)
        elif len(out_train) == 9:
            primal_residuals = out_train[4].mean(axis=0)
            dual_residuals = out_train[5].mean(axis=0)
            dist_opts = out_train[8].mean(axis=0)

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
        iters_df, primal_residuals_df, dual_residuals_df, obj_vals_diff_df, dist_opts_df = df_out

        if not self.skip_startup:
            self.no_learning_accs = write_accuracies_csv(self.accs, iter_losses_mean, train, col, 
                                                         self.no_learning_accs)

        # plot the evaluation iterations
        plot_eval_iters(iters_df, primal_residuals_df,
                        dual_residuals_df, plot_pretrain, obj_vals_diff_df, dist_opts_df, train, 
                        col, self.eval_unrolls, self.train_unrolls)

        # plot the warm-start predictions
        z_all = out_train[2]

        if isinstance(self.l2ws_model, SCSmodel) or isinstance(self.l2ws_model, LASCOSCSmodel):
            out_train[6]
            z_plot = z_all[:, :, :-1] / z_all[:, :, -1:]
        else:
            z_plot = z_all

        plot_warm_starts(self.l2ws_model, self.plot_iterates, z_plot, train, col)

        if self.l2ws_model.algo[:5] == 'lasco':
            plot_lasco_weights(self.l2ws_model.params, col)

        # custom visualize
        if self.has_custom_visualization:
            if self.vis_num > 0:
                self.custom_visualize(z_plot, train, col)

        if self.save_weights_flag:
            self.save_weights()

        return out_train

    def run(self):
        # setup logging and dataframes
        self._init_logging()
        self.setup_dataframes()

        if not self.skip_startup:
            # no learning evaluation
            self.eval_iters_train_and_test('no_train', None)

            # fixed ws evaluation
            # if self.l2ws_model.z_stars_train is not None and self.l2ws_model.algo != 'maml':
            #     self.eval_iters_train_and_test('nearest_neighbor', False)

            # prev sol eval
            if self.prev_sol_eval and self.l2ws_model.z_stars_train is not None:
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


        for window in range(10):
            # update window_indices
            # window_indices = jnp.arange(10 * window, 10 * (window + 1))
            window_indices = jnp.arange(self.train_unrolls * window, self.train_unrolls * (window + 1))

            # update the train inputs
            update_train_inputs_flag = window > 0

            for epoch_batch in range(num_epochs_jit):
                epoch = int(epoch_batch * self.epochs_jit) + window * num_epochs_jit * self.epochs_jit
                print('epoch', epoch)

                if (test_zero and epoch == 0) or (epoch % self.eval_every_x_epochs == 0 and epoch > 0):
                    if update_train_inputs_flag:
                        self.eval_iters_train_and_test(f"train_epoch_{epoch}",self.train_unrolls * window)
                        update_train_inputs_flag = False
                        # self.l2ws_model.initialize_neural_network(self.nn_cfg, 'dont_init')
                    else:
                        self.eval_iters_train_and_test(f"train_epoch_{epoch}", None)

                # setup the permutations
                permutation = setup_permutation(
                    self.key_count, self.l2ws_model.N_train, self.epochs_jit)

                # train the jitted epochs
                curr_params, state, epoch_train_losses, time_train_per_epoch = self.train_jitted_epochs(
                    permutation, epoch, window_indices)
                
                # insert the curr_params into the entire params
                pp = self.l2ws_model.params[0].at[window_indices, :].set(curr_params[0])
                params = [pp]
                # print('curr_params', curr_params)

                # reset the global (params, state)
                self.key_count += 1
                self.l2ws_model.epoch += self.epochs_jit
                self.l2ws_model.params, self.l2ws_model.state = params, state

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

    def train_jitted_epochs(self, permutation, epoch, window_indices):
        """
        train self.epochs_jit at a time
        special case: the first time we call train_batch (i.e. epoch = 0)
        """
        epoch_batch_start_time = time.time()
        loop_size = int(self.l2ws_model.num_batches * self.epochs_jit)
        epoch_train_losses = jnp.zeros(loop_size)
        if epoch == 0:
            # unroll the first iterate so that This allows `init_val` and `body_fun`
            #   below to have the same output type, which is a requirement of
            #   lax.while_loop and lax.scan.
            batch_indices = lax.dynamic_slice(
                permutation, (0,), (self.l2ws_model.batch_size,))
            # batch_indices = lax.dynamic_slice(
            #     jnp.arange(permutation.size), (0,), (self.l2ws_model.batch_size,))

            train_loss_first, params, state = self.l2ws_model.train_batch(
                batch_indices, self.l2ws_model.train_inputs, [self.l2ws_model.params[0][window_indices, :]], self.l2ws_model.state)
            
            epoch_train_losses = epoch_train_losses.at[0].set(train_loss_first)
            start_index = 1
            # self.train_over_epochs_body_simple_fn_jitted = partial(self.train_over_epochs_body_simple_fn, 
            #                                                        window_indices=window_indices)
        else:
            start_index = 0
            params, state = [self.l2ws_model.params[0][window_indices, :]], self.l2ws_model.state
            # self.train_over_epochs_body_simple_fn_jitted = partial(self.train_over_epochs_body_simple_fn, 
            #                                                        window_indices=window_indices)
        train_over_epochs_body_simple_fn_jitted = partial(self.train_over_epochs_body_simple_fn, window_indices=window_indices)

        init_val = epoch_train_losses, self.l2ws_model.train_inputs, params, state, permutation
        val = lax.fori_loop(start_index, loop_size,
                            train_over_epochs_body_simple_fn_jitted, init_val)
        epoch_batch_end_time = time.time()
        time_diff = epoch_batch_end_time - epoch_batch_start_time
        time_train_per_epoch = time_diff / self.epochs_jit
        epoch_train_losses, inputs, params, state, permutation = val

        self.l2ws_model.key = state.iter_num

        return params, state, epoch_train_losses, time_train_per_epoch

    def train_over_epochs_body_simple_fn(self, batch, val, window_indices):
        """
        to be used as the body_fn in lax.fori_loop
        need to call partial for the specific permutation
        """
        train_losses, inputs, params, state, permutation = val
        start_index = batch * self.l2ws_model.batch_size
        batch_indices = lax.dynamic_slice(
            permutation, (start_index,), (self.l2ws_model.batch_size,))
        # batch_indices = lax.dynamic_slice(
        #         jnp.arange(permutation.size), (start_index,), (self.l2ws_model.batch_size,))
        train_loss, params, state = self.l2ws_model.train_batch(
            batch_indices, inputs, params, state)
        # train_loss, params, state = self.l2ws_model.train_batch(
        #     batch_indices, inputs, [params[0][window_indices, :]], state)
        train_losses = train_losses.at[batch].set(train_loss)
        val = train_losses, inputs, params, state, permutation
        return val

    # def train_over_epochs_body_fn(self, batch, val, permutation):
    #     """
    #     to be used as the body_fn in lax.fori_loop
    #     need to call partial for the specific permutation
    #     """
    #     train_losses, inputs, params, state = val
    #     start_index = batch * self.l2ws_model.batch_size
    #     batch_indices = lax.dynamic_slice(
    #         permutation, (start_index,), (self.l2ws_model.batch_size,))
    #     train_loss, params, state = self.l2ws_model.train_batch(
    #         batch_indices, inputs, params, state)
    #     train_losses = train_losses.at[batch].set(train_loss)
    #     val = train_losses, params, state
    #     return val


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
            
            self.l2ws_model.train_inputs = out_train[2][:, new_start_index, :]
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

    def evaluate_only(self, fixed_ws, num, train, col, batch_size):
        tag = 'train' if train else 'test'
        factors = None

        if train:
            z_stars = self.l2ws_model.z_stars_train[:num, :]
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

        inputs = self.get_inputs_for_eval(fixed_ws, num, train, col)

        eval_out = self.l2ws_model.evaluate(
            self.eval_unrolls, inputs, q_mat, z_stars, fixed_ws, factors=factors, tag=tag)
        return eval_out


    def get_inputs_for_eval(self, fixed_ws, num, train, col):
        if fixed_ws:
            if col == 'nearest_neighbor':
                is_osqp = isinstance(self.l2ws_model, OSQPmodel)
                if is_osqp:
                    m, n = self.l2ws_model.m, self.l2ws_model.n
                else:
                    m, n = 0, 0
                inputs = get_nearest_neighbors(is_osqp, self.l2ws_model.train_inputs,
                                               self.l2ws_model.test_inputs, 
                                               self.l2ws_model.z_stars_train,
                                               train, num, m=m, n=n)
            elif col == 'prev_sol':
                # now set the indices (0, num_traj, 2 * num_traj) to zero
                non_last_indices = jnp.mod(jnp.arange(
                    num), self.traj_length) != self.traj_length - 1
                inputs = self.shifted_sol_fn(
                    self.z_stars_test[:num, :][non_last_indices, :])
        else:
            if train:
                inputs = self.l2ws_model.train_inputs[:num, :]
            else:
                inputs = self.l2ws_model.test_inputs[:num, :]
        return inputs

    def setup_dataframes(self):
        self.iters_df_train = create_empty_df(self.eval_unrolls)
        self.iters_df_test = create_empty_df(self.eval_unrolls)

        # primal and dual residuals
        self.primal_residuals_df_train = create_empty_df(self.eval_unrolls)
        self.dual_residuals_df_train = create_empty_df(self.eval_unrolls)
        self.primal_residuals_df_test = create_empty_df(self.eval_unrolls)
        self.dual_residuals_df_test = create_empty_df(self.eval_unrolls)

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
        primal_residuals_df, dual_residuals_df = None, None
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
                primal_residuals_df = self.primal_residuals_df_train
                dual_residuals_df = self.dual_residuals_df_train
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
                primal_residuals_df = self.primal_residuals_df_test
                dual_residuals_df = self.dual_residuals_df_test
            if obj_vals_diff is not None:
                self.obj_vals_diff_df_test[col] = obj_vals_diff
                self.obj_vals_diff_df_test.to_csv('obj_vals_diff_test.csv')
                obj_vals_diff_df = self.obj_vals_diff_df_test
            if dist_opts is not None:
                self.dist_opts_df_test[col] = dist_opts
                self.dist_opts_df_test.to_csv('dist_opts_df_test.csv')
                dist_opts_df = self.dist_opts_df_test

            iters_df = self.iters_df_test

        return iters_df, primal_residuals_df, dual_residuals_df, obj_vals_diff_df, dist_opts_df
