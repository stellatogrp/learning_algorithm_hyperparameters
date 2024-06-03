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


@hydra.main(config_path='configs/unconstrained_qp', config_name='unconstrained_qp_plot.yaml')
def unconstrained_qp_plot_eval_iters(cfg):
    example = 'unconstrained_qp'
    create_journal_results(example, cfg, train=False)


@hydra.main(config_path='configs/mnist', config_name='mnist_plot.yaml')
def mnist_plot_eval_iters(cfg):
    example = 'mnist'
    create_journal_results(example, cfg, train=False)


@hydra.main(config_path='configs/quadcopter', config_name='quadcopter_plot.yaml')
def quadcopter_plot_eval_iters(cfg):
    example = 'quadcopter'
    create_journal_results(example, cfg, train=False)


def create_lasco_results_constrained(example, cfg):
    # for each method, get the data (dist_opt, pr, dr, pr_dr_max)
    # dictionary: method -> list of dist_
    # looks something like: results['l2ws']['pr'] = primal_residuals
    results_dict = populate_results_dict(example, cfg)

    # calculate the accuracies
    accs_dict = populate_accs_dict(results_dict)
    # takes a different form accuracies_dict['lasco'][0.01] = num_iters (it is a single integer)

    # calculate the gains (divide by cold start)
    gains_dict = populate_gains_dict(results_dict)

    # calculate the reduction in iterations for accs
    acc_reductions_dict = populate_acc_reductions_dict(accs_dict)
    # takes a different form accuracies_dict['lasco'][0.01] = reduction (it is a single fraction)

    # do the plotting
    plot_results_dict(results_dict, gains_dict)

    # create the tables (need the accuracies and reductions for this)
    create_acc_reduction_tables(accs_dict, acc_reductions_dict)


def create_acc_reduction_tables(accs_dict, acc_reductions_dict):
    # create pandas dataframe
    df_acc = pd.DataFrame()
    df_percent = pd.DataFrame()
    df_acc_both = pd.DataFrame()

    accs = list(accs_dict['cold_start'].keys())

    # df_acc
    df_acc['accuracies'] = np.array(accs)
    methods = list(accs_dict.keys())
    for i in range(len(methods)):
        method = methods[i]
        curr_accs = np.array([accs_dict[method][acc] for acc in accs])
        df_acc[method] = curr_accs #accs_dict[method][acc]

    # for i in range(len(titles)):
    #     df_acc = update_acc(df_acc, accs, titles[i], metrics_fp[i])
    df_acc.to_csv('accuracies.csv')

    # df_percent
    df_percent['accuracies'] = np.array(accs)
    # no_learning_acc = df_acc['cold_start']
    for i in range(len(methods)):
        method = methods[i]
        curr_reduction = np.array([acc_reductions_dict[method][acc] for acc in accs])
        df_percent[method] = np.round(curr_reduction, decimals=2) 
    # for col in df_acc.columns:
    #     if col != 'accuracies':
    #         val = 1 - df_acc[col] / no_learning_acc
    #         df_percent[col] = np.round(val, decimals=2)
    df_percent.to_csv('iteration_reduction.csv')



def plot_results_dict(results_dict, gains_dict):
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
    for i in range(len(methods)):
        method = methods[i]
        style = titles_2_styles[method]
        marker = titles_2_markers[method]
        color = titles_2_colors[method]
        mark_start = titles_2_marker_starts[method]

        # plot the values
        axes[0, 0].plot(results_dict[method]['pr'], linestyle=style, marker=marker, color=color, 
                                markevery=(2 * mark_start, 2 * 25))
        axes[0, 1].plot(results_dict[method]['dr'], linestyle=style, marker=marker, color=color, 
                                markevery=(2 * mark_start, 2 * 25))
        
        # plot the gains
        axes[1, 0].plot(gains_dict[method]['pr'], linestyle=style, marker=marker, color=color, 
                                markevery=(2 * mark_start, 2 * 25))
        axes[1, 1].plot(gains_dict[method]['dr'], linestyle=style, marker=marker, color=color, 
                                markevery=(2 * mark_start, 2 * 25))

    fig.tight_layout()
    plt.savefig('pr_dr.pdf', bbox_inches='tight')


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


def populate_accs_dict(results_dict):
    gains_dict = {}
    methods = list(results_dict.keys())
    for i in range(len(methods)):
        method = methods[i]
        method_dict = results_dict[method]
        gains_dict[method] = populate_curr_method_acc_dict(method_dict)
    return gains_dict


def populate_curr_method_acc_dict(method_dict):
    accs_dict = {}
    accs = [0.1, 0.01, 0.001, 0.0001]
    pr_dr_maxes = method_dict['pr_dr_max']
    for i in range(len(accs)):
        if pr_dr_maxes.min() < accs[i]:
            num_iters_required = int(np.argmax(pr_dr_maxes < accs[i]))
        else:
            num_iters_required = pr_dr_maxes.size
        accs_dict[accs[i]] = num_iters_required
    return accs_dict


def populate_gains_dict(results_dict):
    cold_start_dict = results_dict['cold_start']
    gains_dict = {}
    methods = list(results_dict.keys())
    for i in range(len(methods)):
        method = methods[i]
        method_dict = results_dict[method]
        gains_dict[method] = populate_curr_method_gain_dict(cold_start_dict, method_dict)
    return gains_dict


def populate_results_dict(example, cfg):
    results_dict = {}
    for method in cfg.methods:
        curr_method_dict = populate_curr_method_dict(method, example, cfg)
        results_dict[method] = curr_method_dict
        # curr_method_dict is a dict of 
        #   {'pr': pr_residuals, 'dr': dr_residuals, 'dist_opt': dist_opts, 'pr_dr_max': pr_dr_maxes}
        # important: nothing to do with reductions or gains here
    return results_dict


def method2col(method):
    if method == 'cold_start':
        col = 'no_learn'
    else:
        col = 'last'
    return col


def populate_curr_method_gain_dict(cold_start_dict, method_dict):
    primal_residuals_gain = cold_start_dict['pr'] / method_dict['pr']
    dual_residuals_gain = cold_start_dict['dr'] / method_dict['dr']
    pr_dr_maxes_gain = cold_start_dict['pr_dr_max'] / method_dict['pr_dr_max']
    dist_opts_gain = cold_start_dict['dist_opts'] / method_dict['dist_opts']

    # populate with pr, dr, pr_dr_max, dist_opt
    curr_method_gain_dict = {'pr': primal_residuals_gain, 
                            'dr': dual_residuals_gain, 
                            'pr_dr_max': pr_dr_maxes_gain,
                            'dist_opts': dist_opts_gain}

    return curr_method_gain_dict


def populate_curr_method_dict(method, example, cfg):
    # get the datetime
    dt = cfg['methods'][method]

    # get the column
    col = method2col(method)

    primal_residuals = recover_data(example, dt, 'primal_residuals_test.csv', col)
    dual_residuals = recover_data(example, dt, 'dual_residuals_test.csv', col)
    pr_dr_maxes = recover_data(example, dt, 'pr_dr_max_test.csv', col)
    dist_opts = recover_data(example, dt, 'dist_opts_df_test.csv', col)

    # populate with pr, dr, pr_dr_max, dist_opt
    curr_method_dict = {'pr': primal_residuals, 
                        'dr': dual_residuals, 
                        'pr_dr_max': pr_dr_maxes,
                        'dist_opts': dist_opts}

    return curr_method_dict


def recover_data(example, dt, filename, col):
    orig_cwd = hydra.utils.get_original_cwd()

    path = f"{orig_cwd}/outputs/{example}/train_outputs/{dt}"
    df = read_csv(f"{path}/{filename}")
    data = get_eval_array(df, col)
    # no_learning_path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/{iters_file}"

    # read the eval iters csv
    # fp_file = 'eval_iters_train.csv' if train else 'eval_iters_test.csv'
    # fp_file = 'iters_compared_train.csv' if train else 'iters_compared_test.csv'
    # fp_df = read_csv(f"{path}/{fp_file}")
    # fp = get_eval_array(fp_df, title)
    # # fp = fp_df[title]

    # # read the primal and dual residausl csv
    # if scs_or_osqp:
    #     pr_file = 'primal_residuals_train.csv' if train else 'primal_residuals_test.csv'
    #     pr_df = read_csv(f"{path}/{pr_file}")
    #     pr = get_eval_array(pr_df, title)
    return data



def create_journal_results(example, cfg, train=False):
    """
    does the following steps

    1. get data 
        1.1 (fixed-point residuals, primal residuals, dual residuals) or 
            (fixed-point residuals, obj_diffs)
        store this in metrics

        1.2 timing data
        store this in time_results

        also need: styles, titles
            styles comes from titles
    2. plot the metrics
    3. create the table for fixed-point residuals
    4. create the table for timing results
    """

    # step 1
    metrics, timing_data, titles = get_all_data(example, cfg, train=train)

    # step 2
    plot_all_metrics(metrics, titles, cfg.eval_iters, vert_lines=True)
    plot_all_metrics(metrics, titles, cfg.eval_iters, vert_lines=False)

    # step 3
    metrics_fp = metrics[0]
    create_fixed_point_residual_table(metrics_fp, titles, cfg.accuracies)



def get_all_data(example, cfg, train=False):
    # setup
    orig_cwd = hydra.utils.get_original_cwd()

    # get the datetimes
    learn_datetimes = cfg.output_datetimes
    if learn_datetimes == []:
        dt = recover_last_datetime(orig_cwd, example, 'train')
        learn_datetimes = [dt]

    cold_start_datetime = cfg.cold_start_datetime
    if cold_start_datetime == '':
        cold_start_datetime = recover_last_datetime(orig_cwd, example, 'train')

    nn_datetime = cfg.nearest_neighbor_datetime
    if nn_datetime == '':
        nn_datetime = recover_last_datetime(orig_cwd, example, 'train')

    metrics_list = []
    timing_data = []

    # check if prev_sol exists
    prev_sol_bool = 'prev_sol_datetime' in cfg.keys()

    benchmarks = ['cold_start', 'nearest_neighbor']
    benchmark_dts = [cold_start_datetime, nn_datetime]
    if prev_sol_bool:
        benchmarks.append('prev_sol')
        benchmark_dts.append(cfg.prev_sol_datetime)

    # for init in ['cold_start', 'nearest_neighbor', 'prev_sol']:
    for i in range(len(benchmarks)):
        init = benchmarks[i]
        datetime = benchmark_dts[i]
        metric, timings = load_data_per_title(example, init, datetime)
        metrics_list.append(metric)
        timing_data.append(timings)

    # learned warm-starts
    k_vals = np.zeros(len(learn_datetimes))
    loss_types = []
    for i in range(len(k_vals)):
        datetime = learn_datetimes[i]
        loss_type = get_loss_type(orig_cwd, example, datetime)
        loss_types.append(loss_type)
        k = get_k(orig_cwd, example, datetime)
        k_vals[i] = k
        metric, timings = load_data_per_title(example, k, datetime)
        metrics_list.append(metric)
        timing_data.append(timings)

    k_vals_new = []
    for i in range(k_vals.size):
        k = k_vals[i]
        new_k = k if k >= 2 else 0
        k_vals_new.append(new_k)
    titles = benchmarks
    for i in range(len(loss_types)):
        loss_type = loss_types[i]
        k = k_vals_new[i]
        titles.append(f"{loss_type}_k{int(k)}")


    # combine metrics
    metrics = [[row[i] for row in metrics_list] for i in range(len(metrics_list[0]))]

    return metrics, timing_data, titles



def load_data_per_title(example, title, datetime, train=False):
    orig_cwd = hydra.utils.get_original_cwd()
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}"
    # no_learning_path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/{iters_file}"

    # read the eval iters csv
    # fp_file = 'eval_iters_train.csv' if train else 'eval_iters_test.csv'
    fp_file = 'iters_compared_train.csv' if train else 'iters_compared_test.csv'
    fp_df = read_csv(f"{path}/{fp_file}")
    fp = get_eval_array(fp_df, title)
    # fp = fp_df[title]

    # read the primal and dual residausl csv
    if scs_or_osqp:
        pr_file = 'primal_residuals_train.csv' if train else 'primal_residuals_test.csv'
        pr_df = read_csv(f"{path}/{pr_file}")
        pr = get_eval_array(pr_df, title)
        # pr = pr_df[title]

        dr_file = 'dual_residuals_train.csv' if train else 'dual_residuals_test.csv'
        dr_df = read_csv(f"{path}/{dr_file}")
        # dr = dr_df[title]
        dr = get_eval_array(dr_df, title)
        metric = [fp, pr, dr]

    # read the obj_diffs csv
    else:
        metric = [fp, fp]
    return metric



def get_eval_array(df, title):
    if title == 'cold_start' or title == 'no_learn':
        data = df['no_train']
    elif title == 'nearest_neighbor':
        data = df['nearest_neighbor']
    elif title == 'prev_sol':
        data = df['prev_sol']
    else:
        # case of the learned warm-start, take the latest column
        data = df.iloc[:, -1]
    return data


def create_fixed_point_residual_table(metrics_fp, titles, accs):
    # create pandas dataframe
    df_acc = pd.DataFrame()
    df_percent = pd.DataFrame()
    df_acc_both = pd.DataFrame()

    # df_acc
    df_acc['accuracies'] = np.array(accs)
    for i in range(len(titles)):
        df_acc = update_acc(df_acc, accs, titles[i], metrics_fp[i])
    df_acc.to_csv('accuracies.csv')

    # df_percent
    df_percent['accuracies'] = np.array(accs)
    no_learning_acc = df_acc['cold_start']
    for col in df_acc.columns:
        if col != 'accuracies':
            val = 1 - df_acc[col] / no_learning_acc
            df_percent[col] = np.round(val, decimals=2)
    df_percent.to_csv('iteration_reduction.csv')

    # save both iterations and fraction reduction in single table
    # df_acc_both['accuracies'] = df_acc['cold_start']
    # df_acc_both['cold_start_iters'] = np.array(accs)
    df_acc_both['accuracies'] = np.array(accs)
    df_acc_both['cold_start_iters'] = df_acc['cold_start']

    for col in df_percent.columns:
        if col != 'accuracies' and col != 'cold_start':
            df_acc_both[col + '_iters'] = df_acc[col]
            df_acc_both[col + '_red'] = df_percent[col]
    df_acc_both.to_csv('accuracies_reduction_both.csv')



def plot_all_metrics(metrics, titles, eval_iters, vert_lines=False):
    """
    metrics is a list of lists

    e.g.
    metrics = [metric_fp, metric_pr, metric_dr]
    metric_fp = [cs, nn-ws, ps-ws, k=5, k=10, ..., k=120]
        where cs is a numpy array
    same for metric_pr and metric_dr

    each metric has a title

    each line within each metric has a style

    note that we do not explicitly care about the k values
        we will manually create the legend in latex later
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12), sharey='row')
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 13), sharey='row')
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12), sharey='row')

    # for i in range(2):

    # yscale
    axes[0, 0].set_yscale('log')
    axes[0, 1].set_yscale('log')

    # x-label
    # axes[0, 0].set_xlabel('evaluation iterations')
    # axes[0, 1].set_xlabel('evaluation iterations')
    fontsize = 40
    title_fontsize = 40
    axes[1, 0].set_xlabel('evaluation iterations', fontsize=fontsize)
    axes[1, 1].set_xlabel('evaluation iterations', fontsize=fontsize)

    # y-label
    # axes[0, 0].set_ylabel('fixed-point residual')
    # axes[1, 0].set_ylabel('gain to cold start')
    axes[0, 0].set_ylabel('test fixed-point residual', fontsize=fontsize)
    axes[1, 0].set_ylabel('test gain to cold start', fontsize=fontsize)

    # axes[0, 0].set_title('fixed-point residual losses')
    # axes[0, 1].set_title('regression losses')
    # axes[1, 0].set_title('fixed-point residual losses')
    # axes[1, 1].set_title('regression losses')
    axes[0, 0].set_title('training with fixed-point residual losses', fontsize=title_fontsize)
    axes[0, 1].set_title('training with regression losses', fontsize=title_fontsize)
    # axes[1, 0].set_title('training with fixed-point residual losses')
    # axes[1, 1].set_title('training with regression losses')

    axes[0, 0].set_xticklabels([])
    axes[0, 1].set_xticklabels([])

    # axes[0, 0].tick_params(axis='y', which='major', pad=15)
    # axes[1, 0].tick_params(axis='y', which='major', pad=15)

    # titles
    # axes[0, 0].set_title('fixed-point residuals with fixed-point residual-based losses')
    # axes[0, 1].set_title('fixed-point residuals with regression-based losses')
    # axes[1, 0].set_title('gain to cold start with fixed-point residual-based losses')
    # axes[1, 1].set_title('gain to cold start with regression-based losses')

    if len(metrics) == 3:
        start = 1
    else:
        start = 0

    # plot the fixed-point residual
    for i in range(1):
        curr_metric = metrics[i]
        for j in range(len(curr_metric)):
            title = titles[j]
            color = titles_2_colors[title]
            style = titles_2_styles[title]
            marker = titles_2_markers[title]
            mark_start = titles_2_marker_starts[title]
            if title[:3] != 'reg':
                axes[0, 0].plot(np.array(curr_metric[j])[start:eval_iters + start], 
                                linestyle=style, marker=marker, color=color, 
                                markevery=(2 * mark_start, 2 * 25))

            if title[:3] != 'obj':
                axes[0, 1].plot(np.array(curr_metric[j])[start:eval_iters + start], 
                                linestyle=style, marker=marker, color=color, 
                                markevery=(2 * mark_start, 2 * 25))


    # plot the gain
    for i in range(1):
        curr_metric = metrics[i]
        for j in range(len(curr_metric)):
            title = titles[j]
            color = titles_2_colors[title]
            style = titles_2_styles[title]
            marker = titles_2_markers[title]
            mark_start = titles_2_marker_starts[title]
            # if j > 0:
            #     gain = cs / np.array(curr_metric[j])[start:eval_iters + start]
            # else:
            #     cs = np.array(curr_metric[j])[start:eval_iters + start]
            if j == 0:
                cs = np.array(curr_metric[j])[start:eval_iters + start]
            else:
                gain = np.clip(cs / np.array(curr_metric[j])[start:eval_iters + start], 
                               a_min=0, a_max=1500)
                if title[:3] != 'reg':
                    axes[1, 0].plot(gain, linestyle=style, marker=marker, color=color, 
                                    markevery=(2 * mark_start, 2 * 25))
                if title[:3] != 'obj':
                    axes[1, 1].plot(gain, linestyle=style, marker=marker, color=color, 
                                    markevery=(2 * mark_start, 2 * 25))

            # if vert_lines:
            #     if title[0] == 'k':
            #         k = int(title[1:])
            #         plt.axvline(k, color=color)
    # plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    
    fig.tight_layout()
    if vert_lines:
        plt.savefig('all_metric_plots_vert.pdf', bbox_inches='tight')
    else:
        plt.savefig('all_metric_plots.pdf', bbox_inches='tight')
    
    plt.clf()




    # now plot the gain on a non-log plot
    # plot the gain
    for i in range(1):
        curr_metric = metrics[i]
        for j in range(len(curr_metric)):
        # for j in range(1):
            title = titles[j]
            # title = 'gain to cold start'
            color = titles_2_colors[title]
            style = titles_2_styles[title]

            if j > 0:
                gain = cs / np.array(curr_metric[j])[start:eval_iters + start]
                plt.plot(gain, linestyle=style, color=color)
            else:
                cs = np.array(curr_metric[j])[start:eval_iters + start]
            if vert_lines:
                if title[0] == 'k':
                    k = int(title[1:])
                    # plt.vlines(k, 0, 1000, color=color)
                    plt.axvline(k, color=color)
    plt.ylabel('gain')
    plt.xlabel('evaluation steps')
    if vert_lines:
        plt.savefig('test_gain_plots_vert.pdf', bbox_inches='tight')
    else:
        plt.savefig('test_gain_plots.pdf', bbox_inches='tight')
    fig.tight_layout()


    # plot the loss and the gain for each loss separately
    for i in range(2):
        # fig_width = 9
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 12), sharey='row') #, sharey=True)

        # for i in range(2):

        # yscale
        axes[0].set_yscale('log')
        # axes[0, 1].set_yscale('log')

        # x-label
        # axes[0].set_xlabel('evaluation iterations', fontsize=fontsize)
        # axes[0, 1].set_xlabel('evaluation iterations')
        axes[1].set_xlabel('evaluation iterations', fontsize=fontsize)
        # axes[1, 1].set_xlabel('evaluation iterations')

        # y-label
        axes[0].set_ylabel('fixed-point residual', fontsize=fontsize)
        axes[1].set_ylabel('gain to cold start', fontsize=fontsize)

        axes[0].set_xticklabels([])

        # axes[0, 0].set_title('fixed-point residual losses')
        # axes[0, 1].set_title('regression losses')
        # axes[1, 0].set_title('fixed-point residual losses')
        # axes[1, 1].set_title('regression losses')

        curr_metric = metrics[0]

        for j in range(len(curr_metric)):
            title = titles[j]
            color = titles_2_colors[title]
            style = titles_2_styles[title]
            marker = titles_2_markers[title]
            mark_start = titles_2_marker_starts[title]
            if title[:3] != 'reg' and i == 0:
                # either obj or baselines
                axes[0].plot(np.array(curr_metric[j])[start:eval_iters + start], linestyle=style, marker=marker, color=color, markevery=(2 * mark_start, 2 * 25))
            if title[:3] != 'obj' and  i == 1:
                # either reg or baselines
                axes[0].plot(np.array(curr_metric[j])[start:eval_iters + start], linestyle=style,   marker=marker, color=color, markevery=(2 * mark_start, 2 * 25))

                

        for j in range(len(curr_metric)):
            title = titles[j]
            color = titles_2_colors[title]
            style = titles_2_styles[title]
            marker = titles_2_markers[title]
            mark_start = titles_2_marker_starts[title]
            if j == 0:
                cs = np.array(curr_metric[j])[start:eval_iters + start]
            gain = np.clip(cs / np.array(curr_metric[j])[start:eval_iters + start], 
                           a_min=0, a_max=1500)
            if title[:3] != 'reg' and i == 0:
                axes[1].plot(gain, linestyle=style, marker=marker, color=color, markevery=(2 * mark_start, 2 * 25))
            if title[:3] != 'obj' and i == 1:
                axes[1].plot(gain, linestyle=style, marker=marker, color=color, markevery=(2 * mark_start, 2 * 25))

        if i == 0:
            plt.savefig('fixed_point_residual_loss.pdf', bbox_inches='tight')
        elif i == 1:
            plt.savefig('regression_loss.pdf', bbox_inches='tight')



def get_loss_type(orig_cwd, example, datetime):
    train_yaml_filename = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/.hydra/config.yaml" # noqa
    with open(train_yaml_filename, "r") as stream:
        try:
            out_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # k = int(out_dict['train_unrolls'])
    loss_type = 'reg' if bool(out_dict['supervised']) else 'obj'
    return loss_type



def get_k(orig_cwd, example, datetime):
    train_yaml_filename = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/.hydra/config.yaml" # noqa
    with open(train_yaml_filename, "r") as stream:
        try:
            out_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    k = int(out_dict['train_unrolls'])
    return k


def get_data(example, datetime, csv_title, eval_iters):
    orig_cwd = hydra.utils.get_original_cwd()
    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/iters_compared.csv"
    df = read_csv(path)
    if csv_title == 'last':
        last_column = df.iloc[:, -1]
    else:
        last_column = df[csv_title]
    return last_column[:eval_iters]


def get_loss_data(example, datetime):
    orig_cwd = hydra.utils.get_original_cwd()

    path = f"{orig_cwd}/outputs/{example}/train_outputs/{datetime}/train_test_results.csv"
    df = read_csv(path)
    # if csv_title == 'last':
    #     last_column = df.iloc[:, -1]
    # else:
    #     last_column = df[csv_title]
    # return last_column[:eval_iters]
    train_losses = df['train_loss']
    test_losses = df['test_loss']
    return train_losses, test_losses



def update_acc(df_acc, accs, col, losses):
    iter_vals = np.zeros(len(accs))
    for i in range(len(accs)):
        if losses.min() < accs[i]:
            iter_vals[i] = int(np.argmax(losses < accs[i]))
        else:
            iter_vals[i] = losses.size
    int_iter_vals = iter_vals.astype(int)
    df_acc[col] = int_iter_vals
    return df_acc




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
    elif sys.argv[1] == 'quadcopter':
        sys.argv[1] = base + 'quadcopter/plots/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        quadcopter_plot_eval_iters()

