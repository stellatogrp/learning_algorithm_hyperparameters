import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pandas import read_csv

from lasco.utils.data_utils import recover_last_datetime

from plotter_lasco_constants import titles_2_colors, titles_2_marker_starts, titles_2_markers, titles_2_styles

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    "font.size": 26,
    # "font.size": 16,
})
cmap = plt.cm.Set1
colors = cmap.colors




@hydra.main(config_path='configs/robust_kalman', config_name='robust_kalman_plot.yaml')
def robust_kalman_plot_eval_iters(cfg):
    example = 'robust_kalman'
    # create_journal_results(example, cfg, train=False)
    create_lasco_results_constrained(example, cfg)


@hydra.main(config_path='configs/maxcut', config_name='maxcut_plot.yaml')
def maxcut_plot_eval_iters(cfg):
    example = 'maxcut'
    create_lasco_results_constrained(example, cfg)


@hydra.main(config_path='configs/ridge_regression', config_name='ridge_regression_plot.yaml')
def ridge_regression_plot_eval_iters(cfg):
    example = 'ridge_regression'
    create_lasco_results_unconstrained(example, cfg)


@hydra.main(config_path='configs/unconstrained_qp', config_name='unconstrained_qp_plot.yaml')
def unconstrained_qp_plot_eval_iters(cfg):
    example = 'unconstrained_qp'
    create_lasco_results_constrained(example, cfg)


@hydra.main(config_path='configs/mnist', config_name='mnist_plot.yaml')
def mnist_plot_eval_iters(cfg):
    example = 'mnist'
    create_lasco_results_constrained(example, cfg)


@hydra.main(config_path='configs/quadcopter', config_name='quadcopter_plot.yaml')
def quadcopter_plot_eval_iters(cfg):
    example = 'quadcopter'
    create_lasco_results_constrained(example, cfg)


def create_lasco_results_unconstrained(example, cfg):
    # for each method, get the data (dist_opt, pr, dr, pr_dr_max)
    # dictionary: method -> list of dist_
    # looks something like: results['l2ws']['pr'] = primal_residuals
    results_dict = populate_results_dict(example, cfg, constrained=False)

    # calculate the accuracies
    accs_dict = populate_accs_dict(results_dict, constrained=False)
    # takes a different form accuracies_dict['lasco'][0.01] = num_iters (it is a single integer)

    # calculate the gains (divide by cold start)
    gains_dict = populate_gains_dict(results_dict, constrained=False)

    # calculate the reduction in iterations for accs
    acc_reductions_dict = populate_acc_reductions_dict(accs_dict)
    # takes a different form accuracies_dict['lasco'][0.01] = reduction (it is a single fraction)

    # do the plotting
    plot_results_dict_unconstrained(results_dict, gains_dict, cfg.num_iters)

    # create the tables (need the accuracies and reductions for this)
    create_acc_reduction_tables(accs_dict, acc_reductions_dict)


def create_lasco_results_constrained(example, cfg):
    # for each method, get the data (dist_opt, pr, dr, pr_dr_max)
    # dictionary: method -> list of dist_
    # looks something like: results['l2ws']['pr'] = primal_residuals
    results_dict = populate_results_dict(example, cfg, constrained=True)

    # calculate the accuracies
    accs_dict = populate_accs_dict(results_dict, constrained=True)
    # takes a different form accuracies_dict['lasco'][0.01] = num_iters (it is a single integer)

    # calculate the gains (divide by cold start)
    gains_dict = populate_gains_dict(results_dict, constrained=True)

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
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12), sharey='row') #, sharey=True)
    axes[0, 0].set_yscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel('iterations')
    axes[1, 1].set_xlabel('iterations')
    axes[0, 0].set_title('primal residuals')
    axes[0, 1].set_title('dual residuals')

    axes[0, 0].set_ylabel('values')
    axes[1, 0].set_ylabel('gain to cold start')

    methods = list(results_dict.keys())
    markevery = int(num_iters / 20)
    for i in range(len(methods)):
        method = methods[i]
        style = titles_2_styles[method]
        marker = titles_2_markers[method]
        color = titles_2_colors[method]
        mark_start = titles_2_marker_starts[method]

        # plot the values
        axes[0, 0].plot(results_dict[method]['pr'][:num_iters], linestyle=style, marker=marker, color=color, 
                                markevery=(mark_start, markevery))
        axes[0, 1].plot(results_dict[method]['dr'][:num_iters], linestyle=style, marker=marker, color=color, 
                                markevery=(mark_start, markevery))
        
        # plot the gains
        axes[1, 0].plot(gains_dict[method]['pr'][:num_iters], linestyle=style, marker=marker, color=color, 
                                markevery=(mark_start, markevery))
        axes[1, 1].plot(gains_dict[method]['dr'][:num_iters], linestyle=style, marker=marker, color=color, 
                                markevery=(mark_start, markevery))

    fig.tight_layout()
    plt.savefig('pr_dr.pdf', bbox_inches='tight')


def plot_results_dict_unconstrained(results_dict, gains_dict, num_iters):
    # plot the primal and dual residuals next to each other
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 12), sharey='row') #, sharey=True)
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('iterations')
    axes[0].set_title('objective suboptimality')

    axes[0].set_ylabel('values')
    axes[1].set_ylabel('gain to cold start')

    methods = list(results_dict.keys())
    markevery = int(num_iters / 20)
    for i in range(len(methods)):
        method = methods[i]
        style = titles_2_styles[method]
        marker = titles_2_markers[method]
        color = titles_2_colors[method]
        mark_start = titles_2_marker_starts[method]

        # plot the values
        axes[0].plot(results_dict[method]['obj_diff'][:num_iters], linestyle=style, marker=marker, color=color, 
                                markevery=(mark_start, markevery))
        
        # plot the gains
        axes[1].plot(gains_dict[method]['obj_diff'][:num_iters], linestyle=style, marker=marker, color=color, 
                                markevery=(mark_start, markevery))

    fig.tight_layout()
    plt.savefig('obj_diff.pdf', bbox_inches='tight')


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


def populate_gains_dict(results_dict, constrained=True):
    cold_start_dict = results_dict['cold_start']
    gains_dict = {}
    methods = list(results_dict.keys())
    for i in range(len(methods)):
        method = methods[i]
        method_dict = results_dict[method]
        gains_dict[method] = populate_curr_method_gain_dict(cold_start_dict, method_dict, constrained)
    return gains_dict


def populate_results_dict(example, cfg, constrained=True):
    results_dict = {}
    for method in cfg.methods:
        curr_method_dict = populate_curr_method_dict(method, example, cfg, constrained)
        results_dict[method] = curr_method_dict
        # curr_method_dict is a dict of 
        #   {'pr': pr_residuals, 'dr': dr_residuals, 'dist_opt': dist_opts, 'pr_dr_max': pr_dr_maxes}
        # important: nothing to do with reductions or gains here
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
    else:
        col = 'last'
    return col


def populate_curr_method_gain_dict(cold_start_dict, method_dict, constrained):
    if constrained:
        primal_residuals_gain = cold_start_dict['pr'] / method_dict['pr']
        dual_residuals_gain = cold_start_dict['dr'] / method_dict['dr']
        pr_dr_maxes_gain = cold_start_dict['pr_dr_max'] / method_dict['pr_dr_max']
        dist_opts_gain = cold_start_dict['dist_opts'] / method_dict['dist_opts']

        # populate with pr, dr, pr_dr_max, dist_opt
        curr_method_gain_dict = {'pr': primal_residuals_gain, 
                                'dr': dual_residuals_gain, 
                                'pr_dr_max': pr_dr_maxes_gain,
                                'dist_opts': dist_opts_gain}
    else:
        curr_method_gain_dict = {'obj_diff': cold_start_dict['obj_diff'] / method_dict['obj_diff']}

    return curr_method_gain_dict


def populate_curr_method_dict(method, example, cfg, constrained):
    # get the datetime
    dt = cfg['methods'][method]

    # get the column
    col = method2col(method)

    if constrained:
        primal_residuals = recover_data(example, dt, 'primal_residuals_test.csv', col)
        dual_residuals = recover_data(example, dt, 'dual_residuals_test.csv', col)
        pr_dr_maxes = recover_data(example, dt, 'pr_dr_max_test.csv', col)
        dist_opts = recover_data(example, dt, 'dist_opts_df_test.csv', col)

        # populate with pr, dr, pr_dr_max, dist_opt
        curr_method_dict = {'pr': primal_residuals, 
                            'dr': dual_residuals, 
                            'pr_dr_max': pr_dr_maxes,
                            'dist_opts': dist_opts}
    else:
        obj_diffs = recover_data(example, dt, 'obj_vals_diff_test.csv', col)
        curr_method_dict = {'obj_diff': obj_diffs}

    return curr_method_dict


def recover_data(example, dt, filename, col):
    orig_cwd = hydra.utils.get_original_cwd()

    path = f"{orig_cwd}/outputs/{example}/train_outputs/{dt}"
    df = read_csv(f"{path}/{filename}")
    data = get_eval_array(df, col)

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
    elif title == 'l2ws':
        data = df.iloc[:, -1]
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
    elif sys.argv[1] == 'quadcopter':
        sys.argv[1] = base + 'quadcopter/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        quadcopter_plot_eval_iters()
