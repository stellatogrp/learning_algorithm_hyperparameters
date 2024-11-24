import csv
import gc
import os
import time
from functools import partial

import hydra
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scs
from jax import lax, vmap
from scipy.sparse import csc_matrix, load_npz
from scipy.spatial import distance_matrix

from lah.algo_steps import (
    create_projection_fn,
    form_osqp_matrix,
    get_psd_sizes,
    unvec_symm,
    vec_symm,
)
from lah.gd_model import GDmodel
from lah.lah_gd_model import LAHGDmodel
from lah.lah_osqp_model import LAHOSQPmodel
from lah.lah_scs_model import LAHSCSmodel
from lah.osqp_model import OSQPmodel
from lah.scs_model import SCSmodel
from lah.utils.generic_utils import count_files_in_directory, sample_plot, setup_permutation

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    "font.size": 16,
})


def plot_eval_iters(iters_df, primal_residuals_df, dual_residuals_df, plot_pretrain,
                        obj_vals_diff_df, dist_opts_df, pr_dr_max_df,
                        train, col, eval_unrolls, train_unrolls):
        yscale = 'log'
        plot_eval_iters_df(
            iters_df, train, col, 'fixed point residual', 'eval_iters', eval_unrolls, train_unrolls,
            yscale=yscale)
        if primal_residuals_df is not None:
            plot_eval_iters_df(primal_residuals_df, train, col,
                                    'primal residual', 'primal_residuals', eval_unrolls, train_unrolls)
            plot_eval_iters_df(dual_residuals_df, train, col,
                                    'dual residual', 'dual_residuals', eval_unrolls, train_unrolls)
            plot_eval_iters_df(pr_dr_max_df, train, col,
                                    'pr dr max', 'pr_dr_max', eval_unrolls, train_unrolls)
        if obj_vals_diff_df is not None:
            plot_eval_iters_df(
                obj_vals_diff_df, train, col, 'obj diff', 'obj_diffs', eval_unrolls, train_unrolls)

        if dist_opts_df is not None:
            plot_eval_iters_df(
                dist_opts_df, train, col, 'opt diff', 'dist_opts', eval_unrolls, train_unrolls)


def plot_eval_iters_df(df, train, col, ylabel, filename,
                       eval_unrolls, train_unrolls,
                           xlabel='evaluation iterations',
                           xvals=None,
                           yscale='log', pac_bayes=False):
    if xvals is None:
        xvals = np.arange(eval_unrolls)
    # plot the cold-start if applicable
    if 'no_train' in df.keys():

        plt.plot(xvals, df['no_train'], 'k-', label='no learning')

    # plot the nearest_neighbor if applicable
    if col != 'no_train' and 'nearest_neighbor' in df.keys():
        plt.plot(xvals, df['nearest_neighbor'],
                    'm-', label='nearest neighbor')

    # plot the prev_sol if applicable
    if col != 'no_train' and col != 'nearest_neighbor' and 'prev_sol' in df.keys():
        plt.plot(xvals, df['prev_sol'], 'c-', label='prev solution')

    # plot the learned warm-start if applicable
    if col != 'no_train' and col != 'pretrain' and col != 'nearest_neighbor' and col != 'prev_sol':  # noqa
        plt.plot(xvals, df[col], label=f"train k={train_unrolls}")
        if pac_bayes:
            plt.plot(xvals, df[col + '_pac_bayes'], label="pac_bayes")
    if yscale == 'log':
        plt.yscale('log')
    # plt.xlabel('evaluation iterations')
    plt.xlabel(xlabel)
    plt.ylabel(f"test {ylabel}")
    plt.legend()
    if train:
        plt.title('train problems')
        plt.savefig(f"{filename}_train.pdf", bbox_inches='tight')
    else:
        plt.title('test problems')
        plt.savefig(f"{filename}_test.pdf", bbox_inches='tight')
    plt.clf()


def plot_train_test_losses(tr_losses_batch, te_losses, num_batches, epochs_jit):
    batch_losses = np.array(tr_losses_batch)
    te_losses = np.array(te_losses)
    num_data_points = batch_losses.size
    epoch_axis = np.arange(num_data_points) / num_batches

    epoch_test_axis = epochs_jit * np.arange(te_losses.size)
    plt.plot(epoch_axis, batch_losses, label='train')
    plt.plot(epoch_test_axis, te_losses, label='test')
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('fixed point residual average')
    plt.legend()
    plt.savefig('losses_over_training.pdf', bbox_inches='tight')
    plt.clf()
    plt.plot(epoch_axis, batch_losses, label='train')

    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('fixed point residual average')
    plt.legend()
    plt.savefig('train_losses_over_training.pdf', bbox_inches='tight')
    plt.clf()


def plot_losses_over_examples(losses_over_examples, train, col, yscalelog=True):
        """
        plots the fixed point residuals over eval steps for each individual problem
        """
        if train:
            loe_folder = 'losses_over_examples_train'
        else:
            loe_folder = 'losses_over_examples_test'
        if not os.path.exists(loe_folder):
            os.mkdir(loe_folder)

        plt.plot(losses_over_examples)

        if yscalelog:
            plt.yscale('log')
        plt.savefig(f"{loe_folder}/losses_{col}_plot.pdf", bbox_inches='tight')
        plt.clf()


def plot_lah_weights(transformed_params, col):
    path = 'lah_weights'
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(f"{path}/{col}"):
        os.mkdir(f"{path}/{col}")

    cmap = plt.cm.Set1
    colors = cmap.colors

    # mean_params = params[0]
    if transformed_params.ndim == 1:
        transformed_params = jnp.expand_dims(transformed_params, axis=1)

    num_params = transformed_params.shape[1]
    for i in range(num_params):
        bars = plt.bar(np.arange(transformed_params[:, i].size), transformed_params[:, i], color=colors[1])
        bars[-1].set_color(colors[0])
        plt.xlabel('iterations')
        plt.ylabel('step size')
        plt.savefig(f"lah_weights/{col}/param_{i}.pdf")
        plt.clf()

    # save to csv
    df = pd.DataFrame(transformed_params)
    df.to_csv(f"lah_weights/{col}/params.csv")


def plot_warm_starts(l2ws_model, plot_iterates, z_all, train, col):
    """
    plots the warm starts for the given method
    """
    if train:
        ws_path = 'warm-starts_train'
    else:
        ws_path = 'warm-starts_test'
    if not os.path.exists(ws_path):
        os.mkdir(ws_path)
    if not os.path.exists(f"{ws_path}/{col}"):
        os.mkdir(f"{ws_path}/{col}")
    for i in range(5):
        # plot for z
        for j in plot_iterates:
            plt.plot(z_all[i, j, :], label=f"prediction_{j}")
        if train:
            plt.plot(l2ws_model.z_stars_train[i, :], label='optimal')
        else:
            plt.plot(l2ws_model.z_stars_test[i, :], label='optimal')
        plt.legend()
        plt.savefig(f"{ws_path}/{col}/prob_{i}_z_ws.pdf")
        plt.clf()

        for j in plot_iterates:
            if isinstance(l2ws_model, OSQPmodel):
                try:
                    plt.plot(z_all[i, j, :l2ws_model.m + l2ws_model.n] -
                                l2ws_model.z_stars_train[i, :],
                                label=f"prediction_{j}")
                except:
                    plt.plot(z_all[i, j, :l2ws_model.m + l2ws_model.n] -
                                l2ws_model.z_stars_train[i, :l2ws_model.m + l2ws_model.n],
                                label=f"prediction_{j}")
            else:
                if train:
                    plt.plot(z_all[i, j, :] - l2ws_model.z_stars_train[i, :],
                                label=f"prediction_{j}")
                else:
                    plt.plot(z_all[i, j, :] - l2ws_model.z_stars_test[i, :],
                                label=f"prediction_{j}")
        plt.legend()
        plt.title('diffs to optimal')
        plt.savefig(f"{ws_path}/{col}/prob_{i}_diffs_z.pdf")
        plt.clf()


def custom_visualize(custom_visualize_fn, iterates_visualize, vis_num, thetas, z_all, 
                     z_stars, z_no_learn, z_nn, z_prev_sol, train, col):
    """
    x_primals has shape [N, eval_iters]
    """
    visualize_path = 'visualize_train' if train else 'visualize_test'

    if not os.path.exists(visualize_path):
        os.mkdir(visualize_path)
    if not os.path.exists(f"{visualize_path}/{col}"):
        os.mkdir(f"{visualize_path}/{col}")

    visual_path = f"{visualize_path}/{col}"

    # call custom visualize fn
    # if train:
    #     z_stars = z_stars_train
    #     thetas = thetas_train
    #     if 'z_nn_train' in dir(self):
    #         z_nn = z_nn_train
    # else:
    #     z_stars = z_stars_test
    #     thetas = thetas_test
    #     if 'z_nn_test' in dir(self):
    #         z_nn = z_nn_test
    #     if 'z_prev_sol_test' in dir(self):
    #         z_prev_sol = z_prev_sol_test
    #     else:
    #         z_prev_sol = None

    # if col == 'no_train':
    #     if train:
    #         z_no_learn_train = z_all
    #     else:
    #         z_no_learn_test = z_all
    # elif col == 'nearest_neighbor':
    #     if train:
    #         z_nn_train = z_all
    #     else:
    #         z_nn_test = z_all
    # elif col == 'prev_sol':
    #     if train:
    #         z_prev_sol_train = z_all
    #     else:
    #         z_prev_sol_test = z_all
    # if train:
    #     z_no_learn = z_no_learn_train
    # else:
    #     z_no_learn = z_no_learn_test

    if train:
        if col != 'nearest_neighbor' and col != 'no_train' and col != 'prev_sol':
            custom_visualize_fn(z_all, z_stars, z_no_learn, z_nn,
                                        thetas, iterates_visualize, visual_path)
    else:
        if col != 'nearest_neighbor' and col != 'no_train': # and col != 'prev_sol':
            if z_prev_sol is None:
                custom_visualize_fn(z_all, z_stars, z_no_learn, z_nn,
                                            thetas, iterates_visualize, visual_path,
                                            num=vis_num)
            else:
                custom_visualize_fn(z_all, z_stars, z_prev_sol, z_nn,
                                            thetas, iterates_visualize, visual_path,
                                            num=vis_num)