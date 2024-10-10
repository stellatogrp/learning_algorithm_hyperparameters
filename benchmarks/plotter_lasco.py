import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pandas import read_csv

from lasco.utils.data_utils import recover_last_datetime

from lasco.examples.robust_kalman import plot_positions_overlay, get_x_kalman_from_x_primal
from lasco.examples.mnist import plot_mult_mnist_img

from plotter_lasco_constants import titles_2_colors, titles_2_marker_starts, titles_2_markers, titles_2_styles

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",   # For talks, use sans-serif
#     "font.size": 30,
#     # "font.size": 16,
# })
import os
import re
cmap = plt.cm.Set1
colors = cmap.colors




@hydra.main(config_path='configs/robust_kalman', config_name='robust_kalman_plot.yaml')
def robust_kalman_plot_eval_iters(cfg):
    example = 'robust_kalman'
    # create_journal_results(example, cfg, train=False)
    create_lasco_results_constrained(example, cfg)
    rkf_vis(example, cfg)


@hydra.main(config_path='configs/maxcut', config_name='maxcut_plot.yaml')
def maxcut_plot_eval_iters(cfg):
    example = 'maxcut'
    create_lasco_results_constrained(example, cfg)


@hydra.main(config_path='configs/ridge_regression', config_name='ridge_regression_plot.yaml')
def ridge_regression_plot_eval_iters(cfg):
    example = 'ridge_regression'
    create_lasco_results_unconstrained(example, cfg)
    plot_step_sizes(example, cfg)


@hydra.main(config_path='configs/logistic_regression', config_name='logistic_regression_plot.yaml')
def logistic_regression_plot_eval_iters(cfg):
    example = 'logistic_regression'
    create_lasco_results_unconstrained(example, cfg)
    plot_step_sizes(example, cfg)


@hydra.main(config_path='configs/lasso', config_name='lasso_plot.yaml')
def lasso_plot_eval_iters(cfg):
    example = 'lasso'
    create_lasco_results_unconstrained(example, cfg)
    plot_step_sizes_lasso(example, cfg)


@hydra.main(config_path='configs/unconstrained_qp', config_name='unconstrained_qp_plot.yaml')
def unconstrained_qp_plot_eval_iters(cfg):
    example = 'unconstrained_qp'
    create_lasco_results_constrained(example, cfg)


@hydra.main(config_path='configs/mnist', config_name='mnist_plot.yaml')
def mnist_plot_eval_iters(cfg):
    example = 'mnist'
    create_lasco_results_constrained(example, cfg)
    mnist_vis(example, cfg)


@hydra.main(config_path='configs/quadcopter', config_name='quadcopter_plot.yaml')
def quadcopter_plot_eval_iters(cfg):
    example = 'quadcopter'
    create_lasco_results_constrained(example, cfg)


def mnist_vis(example, cfg):
    # get the data -- [rkf_lah_data, rk]
    mnist_lah_vis, x_stars, thetas = get_rkf_vis_data(example, cfg.lah_vis_dt)
    mnist_l2ws_vis, _, __ = get_rkf_vis_data(example, cfg.l2ws_vis_dt)
    mnist_lm_vis, _, __ = get_rkf_vis_data(example, cfg.lm_vis_dt)

    figsize = 784
    filename = "quant_mult.pdf"
    indices = np.array([4, 1, 2, 5])
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",   # For talks, use sans-serif
        "font.size": 13,
        # "font.size": 16,
    })
    plt.clf()
    plot_mult_mnist_img(x_stars[indices, :figsize], 
                        thetas[indices, :figsize], 
                        mnist_l2ws_vis[indices, :figsize], 
                        mnist_lm_vis[indices, :figsize], 
                        mnist_lah_vis[indices, :figsize], 
                        filename, figsize=figsize)


def rkf_vis(example, cfg):
    # get the data -- [rkf_lah_data, rk]
    rkf_lah_vis, x_stars, thetas = get_rkf_vis_data(example, cfg.lah_vis_dt)
    rkf_l2ws_vis, _, __ = get_rkf_vis_data(example, cfg.l2ws_vis_dt)
    rkf_lm_vis, _, __ = get_rkf_vis_data(example, cfg.lm_vis_dt)
    # rkf_opt_vis = get_rkf_opt_data(example, cfg.lah_vis_dt)
    # rkf_thetas_vis = get_rkf_thetas_data(example, cfg.lah_vis_dt)

    T = 50
    num = 300
    # iter = 20

    # for i in range(len(cfg.vis_indices)):
    #     index = cfg.vis_indices[i]
    for index in range(200):
        titles = ['optimal solution', 'noisy trajectory']

        y_mat_rotated = np.reshape(thetas[:num, :], (num, T, 2))

        # for i in range(num):
        x_true_kalman = get_x_kalman_from_x_primal(x_stars[index, :], T)
        traj = [x_true_kalman, y_mat_rotated[index, :].T]

        x_kalman_lah = get_x_kalman_from_x_primal(rkf_lah_vis[index,  :], T)
        x_kalman_l2ws = get_x_kalman_from_x_primal(rkf_l2ws_vis[index,  :], T)
        x_kalman_lm = get_x_kalman_from_x_primal(rkf_lm_vis[index,  :], T)
        traj.append(x_kalman_lah)
        traj.append(x_kalman_l2ws)
        traj.append(x_kalman_lm)
        titles.append(f"lah")
        titles.append(f"l2ws")
        titles.append(f"lm")
        plt.clf()
        plot_positions_overlay(traj, titles, filename=f"positions_{index}.pdf", legend=False)
        plot_positions_overlay(traj, titles, filename=f"positions_{index}_leg.pdf", legend=True)
        print('y_mat_rotated', y_mat_rotated[index, :].T)
        print('x_true_kalman', x_true_kalman)
        print('x_kalman_lah', x_kalman_lah)


def get_rkf_vis_data(example, dt):
    orig_cwd = hydra.utils.get_original_cwd()
    dt_path = f"{orig_cwd}/outputs/{example}/train_outputs/{dt}"

    # iterate over all of the ones that start with train_epoch
    directory = f"{dt_path}/visualize_test"
    last_folder = find_last_folder_starting_with(directory, 'train_epoch')

    primals_file = f"{directory}/{last_folder}/x_primals.csv"
    x_primals = read_csv(primals_file, header=None, index_col=0)

    x_stars_file = f"{directory}/{last_folder}/x_stars.csv"
    x_stars = read_csv(x_stars_file, header=None, index_col=0)

    thetas_file = f"{directory}/{last_folder}/thetas.csv"
    thetas = read_csv(thetas_file, header=None, index_col=0)
    return x_primals.to_numpy(), x_stars.to_numpy(), thetas.to_numpy() #[1:, :]


def plot_results_wth_step_sizes(example, cfg, results_dict, gains_dict, num_iters):
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6)) #, sharey='row') #, sharey=True)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    plt.subplots_adjust(wspace=.4)

    # subplot 1: results
    # plot the primal and dual residuals next to each other
    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 12), sharey='row') #, sharey=True)
    axes[0].set_yscale('log')
    # axes[1].set_yscale('log')
    # axes[1].set_xlabel('iterations')
    axes[0].set_title('objective suboptimality')

    # axes[0].set_ylabel('objective suboptimality')
    # axes[1].set_ylabel('gain to vanilla')

    methods = list(results_dict.keys())
    markevery = int(num_iters / 20)
    for i in range(len(methods)):
        method = methods[i]
        style = titles_2_styles[method]
        marker = titles_2_markers[method]
        color = titles_2_colors[method]
        mark_start = titles_2_marker_starts[method]

        if method == 'lm' and 'lm10000' in methods:
            continue
        if method == 'l2ws' and 'l2ws10000' in methods:
            continue

        # plot the values
        axes[0].plot(results_dict[method]['obj_diff'][:num_iters], linestyle=style, marker=marker, color=color, 
                                markevery=(mark_start, markevery))
        
        # # plot the gains
        # axes[1].plot(gains_dict[method]['obj_diff'][:num_iters], linestyle=style, marker=marker, color=color, 
        #                         markevery=(mark_start, markevery))

    # fig.tight_layout()
    # plt.savefig('obj_diff.pdf', bbox_inches='tight')
    # plt.clf()

    # subplot 2: step sizes
    # get the step sizes (for silver and learned)
    step_sizes_dict = get_lasco_gd_step_size(example, cfg)
    lasco_step_sizes = step_sizes_dict['lasco'].to_numpy()[:, 1]

    # get the strongly convex and L-smooth values
    #       can get it from nesterov and no_train
    nesterov_step_size = step_sizes_dict['nesterov'].to_numpy()[0, 1]
    vanilla_step_size = step_sizes_dict['cold_start'].to_numpy()[0, 1]
    smoothness = 1 / nesterov_step_size

    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18, 6), sharey='row') #, sharey=True)
    axes[0].set_xlabel('iterations')
    axes[1].set_xlabel('iterations')
    axes[1].set_title('LAH step sizes')
    # plt.ylabel('step sizes')
    # axes[1, 0].set_ylabel('gain to cold start')

    # plot the bar plot for silver
    cmap = plt.cm.Set1
    colors = cmap.colors
    step_size_iters = cfg.step_size_iters
    # axes[0].bar(np.arange(step_size_iters), silver_step_sizes[:step_size_iters], color=colors[2])
    # axes[0].hlines(2 * nesterov_step_size, 0, step_size_iters, color=colors[3])

    # plot the bar plot for the learned method

    # add in the horizontal lines for lasco
    full_lasco = lasco_step_sizes[-1] * np.ones(step_size_iters)
    num_lasco = lasco_step_sizes.size
    full_lasco[:num_lasco] = lasco_step_sizes[:num_lasco]
    bars = plt.bar(np.arange(step_size_iters), full_lasco, color='0.3') #color=colors[1])
    plt.hlines(2 * nesterov_step_size, 0, step_size_iters, color=colors[7], linewidth=2.5) #color='0.0') #color=colors[3])
    # bars[num_lasco:].set_color(colors[0])
    # Change the color of the bars from num_lasco onward
    for i in range(num_lasco - 1, len(bars)):
        bars[i].set_color('0.6') #colors[0])

    plt.tight_layout()
    plt.savefig('results_with_step_sizes.pdf', bbox_inches='tight')


def plot_step_sizes_lasso(example, cfg):
    plt.figure(figsize=(9, 6))

    # get the step sizes (for silver and learned)
    step_sizes_dict = get_lasco_gd_step_size(example, cfg)
    lasco_step_sizes = step_sizes_dict['lasco'].to_numpy()[:, 1]

    # get the strongly convex and L-smooth values
    #       can get it from nesterov and no_train
    nesterov_step_size = step_sizes_dict['nesterov'].to_numpy()[0, 1]
    vanilla_step_size = step_sizes_dict['cold_start'].to_numpy()[0, 1]
    smoothness = 1 / nesterov_step_size

    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18, 6), sharey='row') #, sharey=True)
    plt.xlabel('iterations')
    plt.title('LAH')
    plt.ylabel('step sizes')
    # axes[1, 0].set_ylabel('gain to cold start')

    # plot the bar plot for silver
    cmap = plt.cm.Set1
    colors = cmap.colors
    step_size_iters = cfg.step_size_iters
    # axes[0].bar(np.arange(step_size_iters), silver_step_sizes[:step_size_iters], color=colors[2])
    # axes[0].hlines(2 * nesterov_step_size, 0, step_size_iters, color=colors[3])

    # plot the bar plot for the learned method

    # add in the horizontal lines for lasco
    full_lasco = lasco_step_sizes[-1] * np.ones(step_size_iters)
    num_lasco = lasco_step_sizes.size
    full_lasco[:num_lasco] = lasco_step_sizes[:num_lasco]
    bars = plt.bar(np.arange(step_size_iters), full_lasco, color='0.2') #color=colors[1])
    plt.hlines(2 * nesterov_step_size, 0, step_size_iters, color='0.5') #color=colors[3])
    # bars[num_lasco:].set_color(colors[0])
    # Change the color of the bars from num_lasco onward
    for i in range(num_lasco - 1, len(bars)):
        bars[i].set_color(colors[0])

    plt.tight_layout()
    plt.savefig('step_sizes.pdf', bbox_inches='tight')


def plot_step_sizes(example, cfg):
    # get the step sizes (for silver and learned)
    step_sizes_dict = get_lasco_gd_step_size(example, cfg)
    silver_step_sizes = step_sizes_dict['silver'].to_numpy()[:, 1]
    lasco_step_sizes = step_sizes_dict['lasco'].to_numpy()[:, 1]

    # get the strongly convex and L-smooth values
    #       can get it from nesterov and no_train
    nesterov_step_size = step_sizes_dict['nesterov'].to_numpy()[0, 1]
    vanilla_step_size = step_sizes_dict['cold_start'].to_numpy()[0, 1]
    smoothness = 1 / nesterov_step_size

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), sharey='row') #, sharey=True)
    axes[0].set_xlabel('iterations')
    axes[1].set_xlabel('iterations')
    axes[0].set_title('silver')
    axes[1].set_title('LAH')
    axes[0].set_ylabel('step sizes')
    # axes[1, 0].set_ylabel('gain to cold start')

    if example == 'logistic_regression':
        axes[0].set_yscale('log')

    # plot the bar plot for silver
    cmap = plt.cm.Set1
    colors = cmap.colors
    step_size_iters = cfg.step_size_iters
    axes[0].bar(np.arange(step_size_iters), silver_step_sizes[:step_size_iters], color=colors[2])
    axes[0].hlines(2 * nesterov_step_size, 0, step_size_iters, color=colors[3])

    # plot the bar plot for the learned method

    # add in the horizontal lines for lasco
    full_lasco = lasco_step_sizes[-1] * np.ones(step_size_iters)
    num_lasco = lasco_step_sizes.size
    full_lasco[:num_lasco] = lasco_step_sizes[:num_lasco]
    bars = axes[1].bar(np.arange(step_size_iters), full_lasco, color=colors[1])
    axes[1].hlines(2 * nesterov_step_size, 0, step_size_iters, color=colors[3])

    # Change the color of the bars from num_lasco onward
    for i in range(num_lasco - 1, len(bars)):
        bars[i].set_color(colors[0])

    plt.tight_layout()
    plt.savefig('step_sizes.pdf', bbox_inches='tight')


    # plot silver + one_step + (two_step + three_step) + 10_step
    lasco_onestep_sizes = step_sizes_dict['lasco_one_step'].to_numpy()[:, 1]
    # lasco_twostep_sizes = step_sizes_dict['lasco_ten_step'].to_numpy()[:, 1]

    if example == 'ridge_regression':
        lasco_twostep_sizes = step_sizes_dict['lasco_two_step'].to_numpy()[:, 1]
        lasco_threestep_sizes = step_sizes_dict['lasco_three_step'].to_numpy()[:, 1]

        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(27, 6), sharey='row') #, sharey=True)
        axes[0].set_xlabel('iterations')
        axes[1].set_xlabel('iterations')
        axes[2].set_xlabel('iterations')
        axes[3].set_xlabel('iterations')
        axes[4].set_xlabel('iterations')
        axes[0].set_title('silver') #, color=colors[2])
        axes[1].set_title('LAH: 1 step at a time') #, color=colors[1])
        axes[2].set_title('LAH: 2 steps at a time') #, color=colors[1])
        axes[3].set_title('LAH: 3 steps at a time') #, color=colors[1])
        axes[4].set_title('LAH: 10 steps at a time') #, color=colors[1])
        # axes[1].set_title('LAH 1 step')
        # axes[2].set_title('LAH 2 step')
        # axes[3].set_title('LAH 10 step')
        axes[0].set_ylabel('step sizes')
        # axes[1, 0].set_ylabel('gain to cold start')


        # plot the bar plot for silver
        cmap = plt.cm.Set1
        colors = cmap.colors
        step_size_iters = cfg.step_size_iters
        axes[0].bar(np.arange(step_size_iters), silver_step_sizes[:step_size_iters],  color='0.3')
        axes[0].hlines(2 * nesterov_step_size, 0, step_size_iters, color=colors[7])

        # plot the bar plot for the learned method

        # add in the horizontal lines for lasco
        full_lasco = lasco_step_sizes[-1] * np.ones(step_size_iters)
        num_lasco = lasco_step_sizes.size
        full_lasco[:num_lasco] = lasco_step_sizes[:num_lasco]
        bars = axes[4].bar(np.arange(step_size_iters), full_lasco, color='0.3')
        axes[4].hlines(2 * nesterov_step_size, 0, step_size_iters, color=colors[7], linewidth=2.5)
        # Change the color of the bars from num_lasco onward
        for i in range(num_lasco - 1, len(bars)):
            bars[i].set_color('0.6')

        # add in the horizontal lines for lasco one step
        full_lasco = lasco_step_sizes[-1] * np.ones(step_size_iters)
        num_lasco = lasco_step_sizes.size
        full_lasco[:num_lasco] = lasco_onestep_sizes[:num_lasco]
        bars = axes[1].bar(np.arange(step_size_iters), full_lasco, color='0.3')
        axes[1].hlines(2 * nesterov_step_size, 0, step_size_iters, color=colors[7], linewidth=2.5)
        # Change the color of the bars from num_lasco onward
        for i in range(num_lasco - 1, len(bars)):
            bars[i].set_color('0.6')

        # add in the horizontal lines for lasco two step
        full_lasco = lasco_step_sizes[-1] * np.ones(step_size_iters)
        num_lasco = lasco_step_sizes.size
        full_lasco[:num_lasco] = lasco_twostep_sizes[:num_lasco]
        full_lasco[50] = full_lasco[-1]
        bars = axes[2].bar(np.arange(step_size_iters), full_lasco, color='0.3')
        axes[2].hlines(2 * nesterov_step_size, 0, step_size_iters, color=colors[7], linewidth=2.5)

        # Change the color of the bars from num_lasco onward
        for i in range(num_lasco - 1, len(bars)):
            bars[i].set_color('0.6')


        # add in the horizontal lines for lasco three step
        full_lasco = lasco_step_sizes[-1] * np.ones(step_size_iters)
        num_lasco = lasco_step_sizes.size
        full_lasco[:num_lasco] = lasco_threestep_sizes[:num_lasco]
        full_lasco[50] = full_lasco[-1]
        bars = axes[3].bar(np.arange(step_size_iters), full_lasco, color='0.3')
        axes[3].hlines(2 * nesterov_step_size, 0, step_size_iters, color=colors[7], linewidth=2.5)

        # Change the color of the bars from num_lasco onward
        for i in range(num_lasco - 2, len(bars)):
            bars[i].set_color('0.6')

        plt.tight_layout()
        plt.savefig('step_sizes_all.pdf', bbox_inches='tight')

    elif example == 'logistic_regression':

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(23, 6), sharey='row') #, sharey=True)
        axes[0].set_xlabel('iterations') #, fontsize=16)
        axes[1].set_xlabel('iterations')
        axes[2].set_xlabel('iterations')
        # axes[3].set_xlabel('iterations')
        axes[0].set_title('silver') #, color=colors[2])
        axes[1].set_title('LAH: 1 step at a time') #, color=colors[1])
        axes[2].set_title('LAH: 10 steps at a time') #, color=colors[1])
        # axes[3].set_title('10 steps at a time', color=colors[1])
        # axes[1].set_title('LAH 1 step')
        # axes[2].set_title('LAH 2 step')
        # axes[3].set_title('LAH 10 step')
        axes[0].set_ylabel('step sizes')
        # axes[1, 0].set_ylabel('gain to cold start')

        if example == 'logistic_regression':
            axes[0].set_yscale('log')

        # plot the bar plot for silver
        cmap = plt.cm.Set1
        colors = cmap.colors
        step_size_iters = cfg.step_size_iters
        axes[0].bar(np.arange(step_size_iters), silver_step_sizes[:step_size_iters], color='0.3')
        axes[0].hlines(2 * nesterov_step_size, 0, step_size_iters, color=colors[7], linewidth=2.5)

        # plot the bar plot for the learned method

        # add in the horizontal lines for lasco
        full_lasco = lasco_step_sizes[-1] * np.ones(step_size_iters)
        num_lasco = lasco_step_sizes.size
        full_lasco[:num_lasco] = lasco_step_sizes[:num_lasco]
        bars = axes[2].bar(np.arange(step_size_iters), full_lasco, color='0.3')
        axes[2].hlines(2 * nesterov_step_size, 0, step_size_iters, color=colors[7], linewidth=2.5)
        # Change the color of the bars from num_lasco onward
        for i in range(num_lasco - 1, len(bars)):
            bars[i].set_color('0.6')

        # add in the horizontal lines for lasco one step
        full_lasco = lasco_step_sizes[-1] * np.ones(step_size_iters)
        num_lasco = lasco_step_sizes.size
        full_lasco[:num_lasco] = lasco_onestep_sizes[:num_lasco]
        full_lasco[100] = full_lasco[101] 
        bars = axes[1].bar(np.arange(step_size_iters), full_lasco, color='0.3')
        axes[1].hlines(2 * nesterov_step_size, 0, step_size_iters, color=colors[7], linewidth=2.5)
        # Change the color of the bars from num_lasco onward
        for i in range(num_lasco - 1, len(bars)):
            bars[i].set_color('0.6')

        # add in the horizontal lines for lasco two step
        # full_lasco = lasco_step_sizes[-1] * np.ones(step_size_iters)
        # num_lasco = lasco_step_sizes.size
        # full_lasco[:num_lasco] = lasco_twostep_sizes[:num_lasco]
        # bars = axes[2].bar(np.arange(step_size_iters), full_lasco, color=colors[1])
        # axes[2].hlines(2 * nesterov_step_size, 0, step_size_iters, color=colors[3])

        # # Change the color of the bars from num_lasco onward
        # for i in range(num_lasco - 1, len(bars)):
        #     bars[i].set_color(colors[0])

        plt.tight_layout()
        plt.savefig('step_sizes_all.pdf', bbox_inches='tight')
    
    



def get_lasco_gd_step_size(example, cfg):
    step_sizes_dict = {}
    for method in cfg.methods:
        dt = cfg.methods[method]
        step_sizes_dict[method] = get_step_sizes(example, dt, method)
    return step_sizes_dict


def get_step_sizes(example, dt, method):
    step_sizes = recover_step_sizes_data(example, dt, method)
    return step_sizes


def recover_step_sizes_data(example, dt, method):
    orig_cwd = hydra.utils.get_original_cwd()
    dt_path = f"{orig_cwd}/outputs/{example}/train_outputs/{dt}"
    df = read_step_size_data(dt_path, method)
    # df = read_csv(f"{path}/{filename}")
    # data = get_eval_array(df, col)
    return df


def read_step_size_data(dt_path, method):
    if method == 'nesterov':
        df = read_csv(f"{dt_path}/lasco_weights/nesterov/params.csv")
    elif method == 'silver':
        df = read_csv(f"{dt_path}/lasco_weights/silver/params.csv")
    elif method == 'cold_start':
        df = read_csv(f"{dt_path}/lasco_weights/no_train/params.csv")
    elif method == 'nearest_neighbor':
        df = read_csv(f"{dt_path}/lasco_weights/nearest_neighbor/params.csv")
    # elif method == 'lasco':
    elif method[:5] == 'lasco':
        # get all of the folder starting with 'train_epoch_...'
        # all_train_epoch_folders = 
        last_folder = find_last_folder_starting_with(f"{dt_path}/lasco_weights", 'train_epoch')
        df = read_csv(f"{dt_path}/lasco_weights/{last_folder}/params.csv") #read_csv(f"{dt_path}/lasco_weights/silver/params.csv")
    else:
        df = None
    return df


# def find_last_folder_starting_with(directory, prefix):
#     # Regular expression to extract the number following the prefix
#     pattern = re.compile(r'^' + re.escape(prefix) + r'(\d+)$')
    
#     max_value = -1
#     last_folder = None

#     # List all directories in the specified directory that match the prefix pattern
#     for name in os.listdir(directory):
#         if os.path.isdir(os.path.join(directory, name)):
#             match = pattern.match(name)
#             if match:
#                 value = int(match.group(1))
#                 if value > max_value:
#                     max_value = value
#                     last_folder = name
#     import pdb
#     pdb.set_trace()
#     return last_folder


def find_last_folder_starting_with(directory, prefix):
    # List all directories in the specified directory that start with the given prefix
    folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and name.startswith(prefix)]
    # Return the last folder alphabetically
    # if folders:
    #     return max(folders)
    # else:
    #     return None
    max_val = 0
    for i in range(len(folders)):
        if 'final' in folders[i]:
            curr_val = int(folders[i][12:-6])
        else:
            curr_val = int(folders[i][12:])
        if curr_val > max_val:
            max_val = curr_val
            last_folder = folders[i]
    return last_folder


def create_lasco_results_unconstrained(example, cfg):
    # for each method, get the data (dist_opt, pr, dr, pr_dr_max)
    # dictionary: method -> list of dist_
    # looks something like: results['l2ws']['pr'] = primal_residuals
    results_dict = populate_results_dict(example, cfg, constrained=False)

    # calculate the accuracies
    accs_dict = populate_accs_dict(results_dict, constrained=False)
    # takes a different form accuracies_dict['lasco'][0.01] = num_iters (it is a single integer)

    # calculate the gains (divide by cold start)
    # gains_dict = populate_gains_dict(results_dict, cfg.num_iters, constrained=False)
    gains_dict = {}

    # calculate the reduction in iterations for accs
    acc_reductions_dict = populate_acc_reductions_dict(accs_dict)
    # takes a different form accuracies_dict['lasco'][0.01] = reduction (it is a single fraction)

    # do the plotting
    plot_results_dict_unconstrained(example, results_dict, gains_dict, cfg.num_iters)

    # create the tables (need the accuracies and reductions for this)
    create_acc_reduction_tables(accs_dict, acc_reductions_dict)

    if example == 'lasso':
        plot_results_wth_step_sizes(example, cfg, results_dict, gains_dict, cfg.num_iters)


def create_lasco_results_constrained(example, cfg):
    # for each method, get the data (dist_opt, pr, dr, pr_dr_max)
    # dictionary: method -> list of dist_
    # looks something like: results['l2ws']['pr'] = primal_residuals
    results_dict = populate_results_dict(example, cfg, constrained=True)

    # calculate the accuracies
    accs_dict = populate_accs_dict(results_dict, constrained=True)
    # takes a different form accuracies_dict['lasco'][0.01] = num_iters (it is a single integer)

    # calculate the gains (divide by cold start)
    gains_dict = populate_gains_dict(results_dict, cfg.num_iters, constrained=True)

    # calculate the reduction in iterations for accs
    acc_reductions_dict = populate_acc_reductions_dict(accs_dict)
    # takes a different form accuracies_dict['lasco'][0.01] = reduction (it is a single fraction)

    # do the plotting
    plot_results_dict_constrained(results_dict, gains_dict, cfg.num_iters)

    # create the tables (need the accuracies and reductions for this)
    create_acc_reduction_tables(accs_dict, acc_reductions_dict)


def create_acc_reduction_tables(accs_dict, acc_reductions_dict):
    # create pandas dataframe
    df_acc = pd.DataFrame()
    df_percent = pd.DataFrame()

    accs = list(accs_dict['cold_start'].keys())

    # df_acc
    df_acc['accuracies'] = np.array(accs)
    methods = list(accs_dict.keys())
    for i in range(len(methods)):
        method = methods[i]
        curr_accs = np.array([accs_dict[method][acc] for acc in accs])
        df_acc[method] = curr_accs
    df_acc.to_csv('accuracies.csv')

    # df_percent
    df_percent['accuracies'] = np.array(accs)
    for i in range(len(methods)):
        method = methods[i]
        curr_reduction = np.array([acc_reductions_dict[method][acc] for acc in accs])
        df_percent[method] = np.round(curr_reduction, decimals=2) 

    df_percent.to_csv('iteration_reduction.csv')



def plot_results_dict_constrained(results_dict, gains_dict, num_iters):
    # plot the primal and dual residuals next to each other
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 12), sharey='row') #, sharey=True)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), sharey='row') #, sharey=True)
    # axes[0, 0].set_yscale('log')
    # axes[1, 0].set_yscale('log')
    # axes[1, 0].set_xlabel('iterations')
    # axes[1, 1].set_xlabel('iterations')
    # axes[0, 0].set_title('primal residuals')
    # axes[0, 1].set_title('dual residuals')

    # axes[0, 0].set_ylabel('residual value')
    # axes[1, 0].set_ylabel('gain to vanilla')

    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[0].set_xlabel('iterations')
    axes[1].set_xlabel('iterations')
    axes[0].set_title('primal residuals')
    axes[1].set_title('dual residuals')

    axes[0].set_ylabel('residual value')

    methods = list(results_dict.keys())
    markevery = int(num_iters / 20)
    for i in range(len(methods)):
        method = methods[i]
        if method == 'lm' and 'lm10000' in methods:
            continue
        if method == 'l2ws' and 'l2ws10000' in methods:
            continue

        style = titles_2_styles[method]
        marker = titles_2_markers[method]
        color = titles_2_colors[method]
        mark_start = titles_2_marker_starts[method]

        # plot the values
        axes[0].plot(results_dict[method]['pr'][:num_iters], linestyle=style, marker=marker, color=color, 
                                markevery=(mark_start, markevery))
        axes[1].plot(results_dict[method]['dr'][:num_iters], linestyle=style, marker=marker, color=color, 
                                markevery=(mark_start, markevery))
        
        # plot the gains
        # axes[1, 0].plot(gains_dict[method]['pr'][:num_iters], linestyle=style, marker=marker, color=color, 
        #                         markevery=(mark_start, markevery))
        # axes[1, 1].plot(gains_dict[method]['dr'][:num_iters], linestyle=style, marker=marker, color=color, 
        #                         markevery=(mark_start, markevery))

    fig.tight_layout()
    plt.savefig('pr_dr.pdf', bbox_inches='tight')


def plot_results_dict_unconstrained(example, results_dict, gains_dict, num_iters):
    # plot the primal and dual residuals next to each other
    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 12), sharey='row') #, sharey=True)
    # axes[0].set_yscale('log')
    # axes[1].set_yscale('log')
    # axes[1].set_xlabel('iterations')
    # axes[0].set_title('objective suboptimality')

    # axes[0].set_ylabel('objective suboptimality')
    # axes[1].set_ylabel('gain to vanilla')

    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 12), sharey='row') #, sharey=True)
    plt.figure(figsize=(12, 6))
    plt.yscale('log')
    # axes[1].set_yscale('log')
    plt.xlabel('iterations')
    plt.title('objective suboptimality')

    if example == 'logistic_regression':
        plt.xscale('log')

    # plt.ylabel('objective suboptimality')
    # axes[1].set_ylabel('gain to vanilla')

    methods = list(results_dict.keys())
    markevery = int(num_iters / 20)
    for i in range(len(methods)):
        method = methods[i]
        style = titles_2_styles[method]
        marker = titles_2_markers[method]
        color = titles_2_colors[method]
        mark_start = titles_2_marker_starts[method]

        if method == 'lm' and 'lm10000' in methods:
            continue
        if method == 'l2ws' and 'l2ws10000' in methods:
            continue

        # plot the values
        if example == 'logistic_regression':
            plt.plot(results_dict[method]['obj_diff'][:num_iters], linestyle=style, marker=marker, color=color,
                     markevery=(mark_start, 0.1))
        else:
            plt.plot(results_dict[method]['obj_diff'][:num_iters], linestyle=style, marker=marker, color=color, 
                                    markevery=(mark_start, markevery))
        
        # plot the gains
        # axes[1].plot(gains_dict[method]['obj_diff'][:num_iters], linestyle=style, marker=marker, color=color, 
        #                         markevery=(mark_start, markevery))

    plt.tight_layout()
    plt.savefig('obj_diff.pdf', bbox_inches='tight')
    plt.clf()


def populate_acc_reductions_dict(accs_dict):
    cold_start_dict = accs_dict['cold_start']
    acc_reductions_dict = {}
    methods = list(accs_dict.keys())
    for i in range(len(methods)):
        method = methods[i]
        method_dict = accs_dict[method]
        acc_reductions_dict[method] = populate_curr_method_acc_reductions_dict(cold_start_dict, 
                                                                               method_dict)
    return acc_reductions_dict


def populate_curr_method_acc_reductions_dict(cold_start_dict, method_dict):
    curr_method_acc_reductions_dict = {}
    accs = [0.1, 0.01, 0.001, 0.0001]
    for i in range(len(accs)):
        curr_method_acc_reductions_dict[accs[i]] = 1 - method_dict[accs[i]] / cold_start_dict[accs[i]]
    return curr_method_acc_reductions_dict


def populate_accs_dict(results_dict, constrained=True):
    gains_dict = {}
    methods = list(results_dict.keys())
    for i in range(len(methods)):
        method = methods[i]
        method_dict = results_dict[method]
        gains_dict[method] = populate_curr_method_acc_dict(method_dict, constrained)
    return gains_dict


def populate_curr_method_acc_dict(method_dict, constrained):
    accs_dict = {}
    accs = [0.1, 0.01, 0.001, 0.0001]
    pr_dr_maxes = method_dict['pr_dr_max'] if constrained else method_dict['obj_diff']
    for i in range(len(accs)):
        if pr_dr_maxes.min() < accs[i]:
            num_iters_required = int(np.argmax(pr_dr_maxes < accs[i]))
        else:
            num_iters_required = pr_dr_maxes.size
        accs_dict[accs[i]] = num_iters_required
    return accs_dict


def populate_gains_dict(results_dict, num_iters, constrained=True):
    cold_start_dict = results_dict['cold_start']
    gains_dict = {}
    methods = list(results_dict.keys())
    for i in range(len(methods)):
        method = methods[i]
        method_dict = results_dict[method]
        gains_dict[method] = populate_curr_method_gain_dict(cold_start_dict, method_dict, constrained, num_iters)
    return gains_dict


def populate_results_dict(example, cfg, constrained=True):
    results_dict = {}
    for method in cfg.methods:
        if method[:2] != 'LB' and method[:2] != 'UB':
            curr_method_dict = populate_curr_method_dict(method, example, cfg, constrained)
            results_dict[method] = curr_method_dict
        # curr_method_dict is a dict of 
        #   {'pr': pr_residuals, 'dr': dr_residuals, 'dist_opt': dist_opts, 'pr_dr_max': pr_dr_maxes}
        # important: nothing to do with reductions or gains here

        # handle the upper and lower bounds for lasco
        else:
            curr_method_dict = populate_curr_method_bound_dict(method, example, cfg, constrained)
            results_dict[method] = curr_method_dict

    return results_dict


def method2col(method):
    if method == 'cold_start':
        col = 'no_learn'
    elif method == 'nearest_neighbor':
        col = 'nearest_neighbor'
    elif method == 'silver':
        col = 'silver'
    elif method == 'nesterov':
        col = 'nesterov'
    elif method == 'conj_grad':
        col = 'conj_grad'
    elif method == 'prev_sol':
        col = 'prev_sol'
    else:
        col = 'last'
    return col


def populate_curr_method_gain_dict(cold_start_dict, method_dict, constrained, num_iters):
    if constrained:
        primal_residuals_gain = np.clip(cold_start_dict['pr'][:num_iters] / method_dict['pr'][:num_iters], a_min=0.001, a_max=1e10)
        dual_residuals_gain = np.clip(cold_start_dict['dr'][:num_iters] / method_dict['dr'][:num_iters], a_min=0.001, a_max=1e10)
        pr_dr_maxes_gain = cold_start_dict['pr_dr_max'][:num_iters] / method_dict['pr_dr_max'][:num_iters]
        # dist_opts_gain = cold_start_dict['dist_opts'] / method_dict['dist_opts']

        # populate with pr, dr, pr_dr_max, dist_opt
        curr_method_gain_dict = {'pr': primal_residuals_gain, 
                                'dr': dual_residuals_gain, 
                                'pr_dr_max': pr_dr_maxes_gain} #,
                                # 'dist_opts': dist_opts_gain}
    else:
        curr_method_gain_dict = {'obj_diff': cold_start_dict['obj_diff'][:num_iters] / method_dict['obj_diff'][:num_iters]}

    return curr_method_gain_dict


def load_frac_solved(example, datetime, acc, upper, title):
    orig_cwd = hydra.utils.get_original_cwd()
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/frac_solved_{title}"

    fp_file = f"tol={acc}_test.csv"
    try:
        df = read_csv(f"{path}/{fp_file}")
    except Exception as e:
        new_acc = 39810.7170553499
        fp_file = f"tol={new_acc}_test.csv"
        df = read_csv(f"{path}/{fp_file}")

    if title is None or True:
        if upper:
            results = df['upper_risk_bound']
        else:
            results = df['lower_risk_bound']
    else:
        results = df[title]
    return results


def get_accs():
    # accuracies = cfg.accuracies
    start = -10  # Start of the log range (log10(10^-5))
    end = 5  # End of the log range (log10(1))
    accuracies = list(np.round(np.logspace(start, end, num=151), 10))
    return accuracies


def get_frac_solved_data_classical(example, dt, upper, title):
    # setup
    # cold_start_datetimes = cfg.cold_start_datetimes
    
    guarantee_results = []

    accuracies = get_accs()
    for acc in accuracies:
        curr_guarantee_results = load_frac_solved(example, dt, acc, upper, title)
        guarantee_results.append(curr_guarantee_results)

    return np.stack(guarantee_results)


def populate_curr_method_bound_dict(method, example, cfg, constrained):
    # get the datetime
    dt = cfg['methods'][method]

    # distinguish between upper and lower bounds
    upper = method[:2] == 'UB'

    # get the column
    col = method2col(method)

    if constrained:
        primal_residuals = recover_data(example, dt, 'primal_residuals_test.csv', col)
        dual_residuals = recover_data(example, dt, 'dual_residuals_test.csv', col)
        pr_dr_maxes = recover_data(example, dt, 'pr_dr_max_test.csv', col)
        # dist_opts = recover_data(example, dt, 'dist_opts_df_test.csv', col)

        # guarantee_results = get_frac_solved_data_classical(example, dt, upper)

        quantile = 100 - float(method[2:])
        # upper = method[:2] == 'UB'
        # accuracies = get_accs()
        # pr_dr_maxes = aggregate_to_quantile_bound(guarantee_results, quantile, accuracies, upper, cfg.num_iters)
        primal_residuals = recover_bound_data(example, dt, upper, quantile, cfg.num_iters, 'primal_residuals')
        dual_residuals = recover_bound_data(example, dt, upper, quantile, cfg.num_iters, 'dual_residuals')
        pr_dr_maxes = recover_bound_data(example, dt, upper, quantile, cfg.num_iters, 'pr_dr_maxes')

        # populate with pr, dr, pr_dr_max, dist_opt
        curr_method_dict = {'pr': np.clip(primal_residuals, a_min=cfg.get('minval', 1e-10), a_max=1e5), 
                            'dr': np.clip(dual_residuals,  a_min=cfg.get('minval', 1e-10), a_max=1e5), 
                            'pr_dr_max': pr_dr_maxes} #,
                            # 'dist_opts': dist_opts}
    else:
        # get the results for all of the tolerances
        # guarantee_results is a list of vectors - each vector is a diff tolerance and gives risk bound over K
        title = 'obj_diffs'
        guarantee_results = get_frac_solved_data_classical(example, dt, upper, title)

        # aggregate into a quantile bound
        quantile = 100 - float(method[2:])
        upper = method[:2] == 'UB'
        accuracies = get_accs()
        obj_diffs = aggregate_to_quantile_bound(guarantee_results, quantile, accuracies, upper, cfg.num_iters)

        # obj_diffs = recover_data(example, dt, 'obj_vals_diff_test.csv', col)
        curr_method_dict = {'obj_diff': obj_diffs}

        # sift through to get the bounds
        
    return curr_method_dict


def recover_bound_data(example, dt, upper, quantile, num_iters, title):
    guarantee_results = get_frac_solved_data_classical(example, dt, upper, title)
    accuracies = get_accs()
    bound_data = aggregate_to_quantile_bound(guarantee_results, quantile, accuracies, upper, num_iters)
    return bound_data



def aggregate_to_quantile_bound(e_stars, quantile, accuracies, upper, cfg_iters):
    eval_iters = e_stars.shape[1]
    # e_stars = get_e_stars(guarantee_results, accuracies, eval_iters)
    
    quantile_curve = np.zeros(eval_iters)
    for k in range(eval_iters):
        if upper:
            where = np.where(e_stars[:, k] < quantile / 100)[0]
            if where.size == 0:
                quantile_curve[k] = max(accuracies)
            else:
                quantile_curve[k] = accuracies[np.min(where)]
        else:
            where = np.where(e_stars[:, k] > quantile / 100)[0]
            if where.size == 0:
                quantile_curve[k] = min(accuracies)
            else:
                quantile_curve[k] = accuracies[np.max(where)]
    return quantile_curve[:cfg_iters]


# def get_e_stars(guarantee_results, accuracies, eval_iters):
#     num_N = len(guarantee_results[0])
#     e_stars = np.zeros((num_N, len(accuracies), eval_iters))
#     for i in range(len(accuracies)):
#         curr_pac_bayes_results = guarantee_results[i]
#         # for j in range(len(curr_pac_bayes_results)):
#         #     curr = curr_pac_bayes_results[j][:eval_iters]
#         #     e_stars[j, i, :curr.size] = curr
#     return e_stars


def populate_curr_method_dict(method, example, cfg, constrained):
    # get the datetime
    dt = cfg['methods'][method]

    # get the column
    col = method2col(method)

    if constrained:
        primal_residuals = recover_data(example, dt, 'primal_residuals_test.csv', col)
        dual_residuals = recover_data(example, dt, 'dual_residuals_test.csv', col)
        pr_dr_maxes = recover_data(example, dt, 'pr_dr_max_test.csv', col)
        # dist_opts = recover_data(example, dt, 'dist_opts_df_test.csv', col)

        # populate with pr, dr, pr_dr_max, dist_opt
        curr_method_dict = {'pr': np.clip(primal_residuals,  a_min=cfg.get('minval', 1e-10), a_max=1e5), 
                            'dr': np.clip(dual_residuals,  a_min=cfg.get('minval', 1e-10), a_max=1e5), 
                            'pr_dr_max': pr_dr_maxes} #,
                            # 'dist_opts': dist_opts}
    else:
        obj_diffs = recover_data(example, dt, 'obj_vals_diff_test.csv', col)
        curr_method_dict = {'obj_diff': obj_diffs}

    return curr_method_dict


def recover_data(example, dt, filename, col, min_val=1e-12):
    orig_cwd = hydra.utils.get_original_cwd()

    path = f"{orig_cwd}/outputs/{example}/train_outputs/{dt}"
    df = read_csv(f"{path}/{filename}")
    data = np.clip(get_eval_array(df, col), a_min=min_val, a_max=1e10)

    return data


def get_eval_array(df, title):
    if title == 'cold_start' or title == 'no_learn':
        data = df['no_train']
    elif title == 'nearest_neighbor':
        data = df['nearest_neighbor']
    elif title == 'silver':
        data = df['silver']
    elif title == 'nesterov':
        data = df['nesterov']
    elif title == 'conj_grad':
        data = df['conj_grad']
    elif title == 'prev_sol':
        data = df['prev_sol']
    elif title == 'l2ws':
        data = df.iloc[:, -1]
    elif title == 'l2ws10000':
        data = df.iloc[:, -1]
    elif title == 'lm' or title == 'lm10000':
        data = df.iloc[:, -1]
    elif title == 'lasco':
        data = df.iloc[:, -2]
    else:
        # case of the learned warm-start, take the latest column
        data = df.iloc[:, -1]
    return data


if __name__ == '__main__':
    if sys.argv[2] == 'cluster':
        base = 'hydra.run.dir=/scratch/gpfs/rajivs/learn2warmstart/outputs/'
    elif sys.argv[2] == 'local':
        base = 'hydra.run.dir=outputs/'
    if sys.argv[1] == 'robust_kalman':
        sys.argv[1] = base + 'robust_kalman/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        robust_kalman_plot_eval_iters()
    elif sys.argv[1] == 'unconstrained_qp':
        sys.argv[1] = base + 'unconstrained_qp/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        unconstrained_qp_plot_eval_iters()
    elif sys.argv[1] == 'mnist':
        sys.argv[1] = base + 'mnist/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        mnist_plot_eval_iters()
    elif sys.argv[1] == 'maxcut':
        sys.argv[1] = base + 'maxcut/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        maxcut_plot_eval_iters()
    elif sys.argv[1] == 'ridge_regression':
        sys.argv[1] = base + 'ridge_regression/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        ridge_regression_plot_eval_iters()
    elif sys.argv[1] == 'logistic_regression':
        sys.argv[1] = base + 'logistic_regression/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        logistic_regression_plot_eval_iters()
    elif sys.argv[1] == 'quadcopter':
        sys.argv[1] = base + 'quadcopter/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        quadcopter_plot_eval_iters()
    elif sys.argv[1] == 'lasso':
        sys.argv[1] = base + 'lasso/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        lasso_plot_eval_iters()
