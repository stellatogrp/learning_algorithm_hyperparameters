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