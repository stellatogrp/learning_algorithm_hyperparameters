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

from lasco.algo_steps import (
    create_projection_fn,
    form_osqp_matrix,
    get_psd_sizes,
    unvec_symm,
    vec_symm,
)
from lasco.gd_model import GDmodel
from lasco.lasco_gd_model import LASCOGDmodel
from lasco.lasco_osqp_model import LASCOOSQPmodel
from lasco.lasco_scs_model import LASCOSCSmodel
from lasco.osqp_model import OSQPmodel
from lasco.scs_model import SCSmodel
from lasco.utils.generic_utils import count_files_in_directory, sample_plot, setup_permutation


def setup_scs_opt_sols(jnp_load_obj, N_train, N):
    if 'x_stars' in jnp_load_obj.keys():
        x_stars = jnp_load_obj['x_stars']
        y_stars = jnp_load_obj['y_stars']
        s_stars = jnp_load_obj['s_stars']
        z_stars = jnp.hstack([x_stars, y_stars + s_stars])
        x_stars_train = x_stars[:N_train, :]
        y_stars_train = y_stars[:N_train, :]

        x_stars_train = x_stars[:N_train, :]
        y_stars_train = y_stars[:N_train, :]

        z_stars_train = z_stars[:N_train, :]
        x_stars_test = x_stars[N_train:N, :]
        y_stars_test = y_stars[N_train:N, :]
        z_stars_test = z_stars[N_train:N, :]
        m, n = y_stars_train.shape[1], x_stars_train[0, :].size
    else:
        x_stars_train, x_stars_test = None, None
        y_stars_train, y_stars_test = None, None
        z_stars_train, z_stars_test = None, None
        m, n = int(jnp_load_obj['m']), int(jnp_load_obj['n'])
    opt_train_sols = (x_stars_train, y_stars_train, z_stars_train)
    opt_test_sols = (x_stars_test, y_stars_test, z_stars_test)
    return opt_train_sols, opt_test_sols, m, n


def get_nearest_neighbors(is_osqp, train_inputs, test_inputs, z_stars_train, train, num, m=0, n=0):
    if train:
        distances = distance_matrix(
            np.array(train_inputs[:num, :]),
            np.array(train_inputs))
    else:
        distances = distance_matrix(
            np.array(test_inputs[:num, :]),
            np.array(train_inputs))
    indices = np.argmin(distances, axis=1)
    plt.plot(indices)
    if train:
        plt.savefig("indices_train_plot.pdf", bbox_inches='tight')
    else:
        plt.savefig("indices_train_plot.pdf", bbox_inches='tight')
    plt.clf()
    if is_osqp:
        return z_stars_train[indices, :m + n]
    return z_stars_train[indices, :]


def plot_samples(num_plot, thetas, train_inputs, z_stars):
        sample_plot(thetas, 'theta', num_plot)
        sample_plot(train_inputs, 'input', num_plot)
        if z_stars is not None:
            sample_plot(z_stars, 'z_stars', num_plot)

def plot_samples_scs(num_plot, thetas, train_inputs, x_stars, y_stars, z_stars):
    sample_plot(thetas, 'theta', num_plot)
    sample_plot(train_inputs, 'input', num_plot)
    if x_stars is not None:
        sample_plot(x_stars, 'x_stars', num_plot)
        sample_plot(y_stars, 'y_stars', num_plot)
        sample_plot(z_stars, 'z_stars', num_plot)

def stack_tuples(tuples_list):
    result = []
    num_tuples = len(tuples_list)
    tuple_length = len(tuples_list[0])

    for i in range(tuple_length):
        stacked_entry = []
        for j in range(num_tuples):
            stacked_entry.append(tuples_list[j][i])
        # result.append(tuple(stacked_entry))
        if tuples_list[j][i] is None:
            result.append(None)
        elif tuples_list[j][i].ndim == 2:
            result.append(jnp.vstack(stacked_entry))
        elif tuples_list[j][i].ndim == 1:
            result.append(jnp.hstack(stacked_entry))
        # elif tuples_list[j][i].ndim == 3 and i == 0:
        #     result.append(jnp.vstack(stacked_entry))
        elif tuples_list[j][i].ndim == 3:
            result.append(jnp.vstack(stacked_entry))
    return result

def normalize_inputs_fn(normalize_inputs, thetas, N_train, N_test):
    # normalize the inputs if the option is on
    N = N_train + N_test
    if normalize_inputs:
        col_sums = thetas.mean(axis=0)
        std_devs = thetas.std(axis=0)
        inputs_normalized = (thetas - col_sums) / std_devs
        inputs = jnp.array(inputs_normalized)

        # save the col_sums and std deviations
        normalize_col_sums = col_sums
        normalize_std_dev = std_devs
    else:
        inputs = jnp.array(thetas)
        normalize_col_sums, normalize_std_dev = 0, 0
    train_inputs = inputs[:N_train, :]
    test_inputs = inputs[N_train:N, :]

    return train_inputs, test_inputs, normalize_col_sums, normalize_std_dev