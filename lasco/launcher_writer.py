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


def update_percentiles(percentiles_df_list, percentiles, losses, train, col):
    path = 'percentiles'
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(len(percentiles)):
        if train:
            filename = f"{path}/train_{percentiles[i]}.csv"
            # curr_percentile = np.percentile(
            #     losses, percentiles[i], axis=0)
            # percentiles_df_list[i][col] = curr_percentile
            # percentiles_df_list[i].to_csv(filename)
        else:
            filename = f"{path}/test_{percentiles[i]}.csv"
        curr_percentile = np.percentile(
            losses, percentiles[i], axis=0)
        percentiles_df_list[i][col] = curr_percentile
        percentiles_df_list[i].to_csv(filename)
    return percentiles_df_list


def write_accuracies_csv(accs, losses, train, col, no_learning_accs):
    df_acc = pd.DataFrame()
    df_acc['accuracies'] = np.array(accs)

    if train:
        accs_path = 'accuracies_train'
    else:
        accs_path = 'accuracies_test'
    if not os.path.exists(accs_path):
        os.mkdir(accs_path)
    if not os.path.exists(f"{accs_path}/{col}"):
        os.mkdir(f"{accs_path}/{col}")

    # accuracies
    iter_vals = np.zeros(len(accs))
    for i in range(len(accs)):
        if losses.min() < accs[i]:
            iter_vals[i] = int(np.argmax(losses < accs[i]))
        else:
            iter_vals[i] = losses.size
    int_iter_vals = iter_vals.astype(int)
    df_acc[col] = int_iter_vals
    df_acc.to_csv(f"{accs_path}/{col}/accuracies.csv")

    # save no learning accuracies
    # if not hasattr(self, 'no_learning_accs'):  # col == 'no_train':
    #     no_learning_accs = int_iter_vals
    if no_learning_accs is None:
        no_learning_accs = int_iter_vals

    # percent reduction
    df_percent = pd.DataFrame()
    df_percent['accuracies'] = np.array(accs)

    for col in df_acc.columns:
        if col != 'accuracies':
            val = 1 - df_acc[col] / no_learning_accs
            df_percent[col] = np.round(val, decimals=2)
    df_percent.to_csv(f"{accs_path}/{col}/reduction.csv")
    return no_learning_accs


def write_train_results(writer, logf, tr_losses_batch, loop_size, prev_batches, epoch_train_losses,
                            time_train_per_epoch):
    for batch in range(loop_size):
        start_window = prev_batches - 10 + batch
        end_window = prev_batches + batch
        last10 = np.array(tr_losses_batch[start_window:end_window])
        moving_avg = last10.mean()
        writer.writerow({
            'train_loss': epoch_train_losses[batch],
            'moving_avg_train': moving_avg,
            'time_train_per_epoch': time_train_per_epoch
        })
        logf.flush()
    return writer, logf