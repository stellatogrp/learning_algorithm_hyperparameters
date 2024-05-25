import time

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scs
from scipy.sparse import csc_matrix

# from lasco.algo_steps import (
#     create_M,
#     create_projection_fn,
#     get_scale_vec,
#     k_steps_eval_scs,
#     k_steps_train_scs,
#     lin_sys_solve,
# )
from lasco.lasco_gd_model import LASCOGDmodel


def test_train_vs_eval_gd():
    # get problem setup
    N_train = 10
    N_test = 10
    N = N_train + N_test
    n = 10
    p = jnp.arange(1, n + 1)
    P = jnp.diag(p) * 1.0
    q_mat = np.random.normal(size=(N, n))
    q_mat_train = q_mat[:N_train, :]
    q_mat_test = q_mat[N_train:, :]

    P_inv = jnp.diag(1 / p)
    z_stars = (-P_inv @ q_mat.T).T
    z_stars_train = z_stars[:N_train, :]
    z_stars_test = z_stars[N_train:, :]

    # create the lasco model
    gd_step = 0.01
    input_dict = dict(algorithm='lasco_gd',
                          c_mat_train=q_mat_train,
                          c_mat_test=q_mat_test,
                          gd_step=gd_step,
                          P=P
                          )
    lasco_gd_model = LASCOGDmodel(
                 train_unrolls=5,
                 train_inputs=q_mat_train,
                 test_inputs=q_mat_test,
                 jit=True,
                 eval_unrolls=20,
                 z_stars_train=z_stars_train,
                 z_stars_test=z_stars_test,
                 loss_method='fixed_k',
                 algo_dict=input_dict)

    # initialize from zero
    z0 = jnp.zeros((N_train, n))
    _, out_from_zero, __ = lasco_gd_model.static_eval(10, z0, q_mat_train, z_stars_train, 0)
    z1_from_zero = out_from_zero[2][:, 1, :]
    z2_from_zero = out_from_zero[2][:, 2, :]
    z10_from_zero = out_from_zero[2][:, 10, :]

    # initialize from z1_from_zero
    _, out_from_old_z1, __ = lasco_gd_model.static_eval(10, z1_from_zero, q_mat_train, z_stars_train, 0)
    z1_from_old_z1 =  out_from_old_z1[2][:, 1, :]

    assert(jnp.linalg.norm(z2_from_zero - z1_from_old_z1) < 1e-8)

    # evaluate with the train method
    out_train = lasco_gd_model.loss_fn_train(lasco_gd_model.params,
                                             z0, 
                                             q_mat_train,
                                             10, 
                                             z_stars_train,
                                             0)
    assert(jnp.abs(out_train - jnp.linalg.norm(z10_from_zero - z_stars_train, axis=1).mean()) < 1e-8)


