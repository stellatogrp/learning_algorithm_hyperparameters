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
from lasco.lasco_scs_model import LASCOSCSmodel


def test_train_vs_eval_gd():
    # get problem setup
    N_train = 10
    N_test = 10
    N = N_train + N_test
    n = 10
    p = jnp.arange(1, n + 1)
    P = jnp.diag(p)
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


    # train on the first 5 steps
    num_iters = 100
    params, state = [lasco_gd_model.params[0][:5, :]], lasco_gd_model.state
    batch_indices = jnp.arange(N_train)
    losses = np.zeros(num_iters)
    inputs = z0
    for i in range(num_iters):
        loss, params, state = lasco_gd_model.train_batch(batch_indices, inputs, params, state) 
        losses[i] = loss
    pp = lasco_gd_model.params[0]
    pp = pp.at[:5, :].set(params[0])
    lasco_gd_model.params = [pp]
    lasco_gd_model.state = state
    assert losses[0] > .1
    assert losses[-1] < losses[0] * .995

    # evaluate from zero
    _, out_from_zero, __ = lasco_gd_model.static_eval(10, z0, q_mat_train, z_stars_train, 0)
    z5_from_zero = out_from_zero[2][:, 5, :]
    z10_from_zero = out_from_zero[2][:, 10, :]
    out_train = lasco_gd_model.loss_fn_train(lasco_gd_model.params,
                                             z0, 
                                             q_mat_train,
                                             5, 
                                             z_stars_train,
                                             0)
    assert(jnp.abs(out_train - jnp.linalg.norm(z5_from_zero - z_stars_train, axis=1).mean()) < 1e-8)

    # shift the params and evaluate from z5
    lasco_gd_model.params = [lasco_gd_model.params[0][5:, :]]
    _, out_from_z5, __ = lasco_gd_model.static_eval(10, z5_from_zero, q_mat_train, z_stars_train, 0)
    z5_5_from_z5 = out_from_z5[2][:, 5, :]

    assert(jnp.linalg.norm(z5_5_from_z5 - z10_from_zero) < 1e-8)

    # evaluate with the training fn
    out_train = lasco_gd_model.loss_fn_train(lasco_gd_model.params,
                                             z5_from_zero, 
                                             q_mat_train,
                                             5, 
                                             z_stars_train,
                                             0)
    assert(jnp.abs(out_train - jnp.linalg.norm(z5_5_from_z5 - z_stars_train, axis=1).mean()) < 1e-8)


def test_train_vs_eval_maxcut():
    # get problem setup
    N_train = 10
    N_test = 10
    N = N_train + N_test
    n = 10
    p = jnp.arange(1, n + 1)
    P = jnp.diag(p)
    q_mat = np.random.normal(size=(N, n))
    q_mat_train = q_mat[:N_train, :]
    q_mat_test = q_mat[N_train:, :]

    P_inv = jnp.diag(1 / p)
    z_stars = (-P_inv @ q_mat.T).T
    z_stars_train = z_stars[:N_train, :]
    z_stars_test = z_stars[N_train:, :]

    # create the lasco model
    gd_step = 0.01
    input_dict = dict(algorithm='lasco_scs',
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


    # train on the first 5 steps
    num_iters = 100
    params, state = [lasco_gd_model.params[0][:5, :]], lasco_gd_model.state
    batch_indices = jnp.arange(N_train)
    losses = np.zeros(num_iters)
    inputs = z0
    for i in range(num_iters):
        loss, params, state = lasco_gd_model.train_batch(batch_indices, inputs, params, state) 
        losses[i] = loss
    pp = lasco_gd_model.params[0]
    pp = pp.at[:5, :].set(params[0])
    lasco_gd_model.params = [pp]
    lasco_gd_model.state = state
    assert losses[0] > .1
    assert losses[-1] < losses[0] * .995

    # evaluate from zero
    _, out_from_zero, __ = lasco_gd_model.static_eval(10, z0, q_mat_train, z_stars_train, 0)
    z5_from_zero = out_from_zero[2][:, 5, :]
    z10_from_zero = out_from_zero[2][:, 10, :]
    out_train = lasco_gd_model.loss_fn_train(lasco_gd_model.params,
                                             z0, 
                                             q_mat_train,
                                             5, 
                                             z_stars_train,
                                             0)
    assert(jnp.abs(out_train - jnp.linalg.norm(z5_from_zero - z_stars_train, axis=1).mean()) < 1e-8)

    # shift the params and evaluate from z5
    lasco_gd_model.params = [lasco_gd_model.params[0][5:, :]]
    _, out_from_z5, __ = lasco_gd_model.static_eval(10, z5_from_zero, q_mat_train, z_stars_train, 0)
    z5_5_from_z5 = out_from_z5[2][:, 5, :]

    assert(jnp.linalg.norm(z5_5_from_z5 - z10_from_zero) < 1e-8)

    # evaluate with the training fn
    out_train = lasco_gd_model.loss_fn_train(lasco_gd_model.params,
                                             z5_from_zero, 
                                             q_mat_train,
                                             5, 
                                             z_stars_train,
                                             0)
    assert(jnp.abs(out_train - jnp.linalg.norm(z5_5_from_z5 - z_stars_train, axis=1).mean()) < 1e-8)
