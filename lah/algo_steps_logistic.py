from functools import partial

import cvxpy as cp
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, lax, vmap
from jax.tree_util import tree_map

from lah.utils.generic_utils import python_fori_loop, unvec_symm, vec_symm


def k_steps_eval_lah_logisticgd(k, z0, q, params, num_points, safeguard_step, supervised, z_star, jit, safeguard=True):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    if safeguard:
        fp_eval_partial = partial(fp_eval_lah_logisticgd_safeguard,
                              supervised=supervised,
                              z_star=z_star,
                              X=X,
                              y=y,
                              safeguard_step=safeguard_step,
                              gd_steps=params
                              )
    else:
        fp_eval_partial = partial(fp_eval_lah_logisticgd,
                                supervised=supervised,
                                z_star=z_star,
                                X=X,
                                y=y,
                                gd_steps=params
                                )
    # nesterov
    fp_eval_nesterov_partial = partial(fp_eval_nesterov_logisticgd,
                            supervised=supervised,
                            z_star=z_star,
                            X=X,
                            y=y,
                            gd_steps=safeguard_step #params
                            )
    
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    val = z0, iter_losses, z_all, obj_diffs

    switch_2_nesterov = False

    if switch_2_nesterov:
        # run learned for 100
        start_iter = 0
        if jit:
            out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
        else:
            out = python_fori_loop(start_iter, k, fp_eval_partial, val)
        z_final, iter_losses, z_all, obj_diffs = out

        # run nesterov for the rest
        start_iter = 0
        iter_losses_nesterov = jnp.zeros(k - 100)
        z_all_nesterov = jnp.zeros((k - 100, z0.size))
        obj_diffs_nesterov = jnp.zeros(k - 100)
        val = z_final, z_final, 0, iter_losses_nesterov, z_all_nesterov, obj_diffs_nesterov
        if jit:
            out_nesterov = lax.fori_loop(start_iter, k - 100, fp_eval_nesterov_partial, val)
        else:
            out_nesterov = python_fori_loop(start_iter, k - 100, fp_eval_nesterov_partial, val)
        z_final_nesterov, y_final_nesterov, t_final_nesterov, iter_losses_nesterov, z_all_nesterov, obj_diffs_nesterov = out_nesterov

        # z_final, y_final, t_final, iter_losses, z_all, obj_diffs

        # stitch everything together
        z_final = z_final_nesterov

        z_all = z_all.at[100:, :].set(z_all_nesterov)
        iter_losses = iter_losses.at[100:].set(iter_losses_nesterov)
        obj_diffs = obj_diffs.at[100:].set(obj_diffs_nesterov)

        z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    else:
        z_all = jnp.zeros((k, z0.size))
        obj_diffs = jnp.zeros(k)
        safeguard = False
        prev_obj = 1e10
        val = z0, iter_losses, z_all, obj_diffs, safeguard, prev_obj
        start_iter = 0
        if jit:
            out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
        else:
            out = python_fori_loop(start_iter, k, fp_eval_partial, val)
        z_final, iter_losses, z_all, obj_diffs, safeguard, prev_obj = out
        z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def k_steps_train_lah_logisticgd(k, z0, q, params, num_points, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    fp_train_partial = partial(fp_train_lah_logisticgd,
                               supervised=supervised,
                               z_star=z_star,
                               X=X,
                               y=y,
                               gd_steps=params
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses

def fp_eval_lah_logisticgd(i, val, supervised, z_star, X, y, gd_steps):
    z, loss_vec, z_all, obj_diffs = val
    z_next = fixed_point_logisticgd(z, X, y, gd_steps[i])
    diff = jnp.linalg.norm(z - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    w, b = z[:-1], z[-1]
    obj = compute_loss(y, sigmoid(X @ w + b))
    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y, sigmoid(X @ w_star + b_star))


    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def fp_eval_lah_logisticgd_safeguard(i, val, supervised, z_star, X, y, safeguard_step, gd_steps):
    # z, loss_vec, z_all, obj_diffs = val
    z, loss_vec, z_all, obj_diffs, safeguard, prev_obj = val
    z_next = fixed_point_logisticgd(z, X, y, gd_steps[i])
    diff = jnp.linalg.norm(z - z_star)
    loss_vec = loss_vec.at[i].set(diff)

    w, b = z[:-1], z[-1]
    obj = compute_loss(y, sigmoid(X @ w + b))

    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y, sigmoid(X @ w_star + b_star))

    w_next, b_next = z_next[:-1], z_next[-1]
    next_obj = compute_loss(y, sigmoid(X @ w_next + b_next))

    # z_next = lax.cond(next_obj < obj, lambda _: z_next, lambda _: fixed_point_logisticgd(z, X, y, safeguard_step), operand=None)

    next_safeguard = lax.cond(next_obj - opt_obj > 20 * (obj - opt_obj), lambda _: True, lambda _: safeguard, operand=None)
    # next_safeguard = lax.cond(next_obj - opt_obj > 10 * prev_obj, lambda _: True, lambda _: safeguard, operand=None)
    z_next_final = lax.cond(next_safeguard, lambda _: fixed_point_logisticgd(z, X, y, safeguard_step), lambda _: z_next, operand=None)


    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    # return z_next, loss_vec, z_all, obj_diffs
    return z_next_final, loss_vec, z_all, obj_diffs, next_safeguard, obj - opt_obj


def fp_train_lah_logisticgd(i, val, supervised, z_star, X, y, gd_steps):
    z, loss_vec = val
    z_next = fixed_point_logisticgd(z, X, y, gd_steps[i])
    diff = jnp.linalg.norm(z_next - z_star) ** 2
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fixed_point_logisticgd(z, X, y, gd_step):
    w, b = z[:-1], z[-1]
    logits = jnp.clip(X @ w + b, -20, 20)
    y_hat = sigmoid(logits)
    # y_hat = sigmoid(X @ w + b)
    
    dw, db = compute_gradient(X, y, y_hat)
    w_next = w - gd_step * dw #(dw + .01 * w)
    b_next = b - gd_step * db #(db + .01 * b)
    z_next = jnp.hstack([w_next, b_next])
    return z_next


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def compute_gradient(X, y, y_hat):
    m = y.shape[0]
    dw = 1/m * jnp.dot(X.T, (y_hat - y))
    db = 1/m * jnp.sum(y_hat - y)
    return dw, db


def compute_loss(y, y_hat):
    m = y.shape[0]
    return -1/m * (jnp.dot(y, jnp.log(1e-6 + y_hat)) + jnp.dot((1 - y), jnp.log(1 + 1e-6 - y_hat)))


def k_steps_eval_logisticgd(k, z0, q, gd_step, num_points, supervised, z_star, jit, safeguard=True):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    # if safeguard:
    #     fp_eval_partial = partial(fp_eval_logisticgd_safeguard,
    #                           supervised=supervised,
    #                           z_star=z_star,
    #                           X=X,
    #                           y=y,
    #                           safeguard_step=safeguard_step,
    #                           gd_steps=params
    #                           )
    # else:
    fp_eval_partial = partial(fp_eval_logisticgd,
                            supervised=supervised,
                            z_star=z_star,
                            X=X,
                            y=y,
                            gd_step=gd_step
                            )
    
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    val = z0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def k_steps_train_logisticgd(k, z0, q, gd_step, num_points, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    fp_train_partial = partial(fp_train_logisticgd,
                               supervised=supervised,
                               z_star=z_star,
                               X=X,
                               y=y,
                               gd_step=gd_step
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def fp_eval_logisticgd(i, val, supervised, z_star, X, y, gd_step):
    z, loss_vec, z_all, obj_diffs = val
    z_next = fixed_point_logisticgd(z, X, y, gd_step)
    diff = jnp.linalg.norm(z - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    w, b = z[:-1], z[-1]
    obj = compute_loss(y, sigmoid(X @ w + b))
    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y, sigmoid(X @ w_star + b_star))


    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def fp_train_logisticgd(i, val, supervised, z_star, X, y, gd_step):
    z, loss_vec = val
    z_next = fixed_point_logisticgd(z, X, y, gd_step)
    diff = jnp.linalg.norm(z_next - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def k_steps_eval_lm_logisticgd(k, z0, q, params, num_points, supervised, z_star, jit, safeguard=False):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    if safeguard:
        fp_eval_partial = partial(fp_eval_lm_logisticgd_safeguard,
                              supervised=supervised,
                              z_star=z_star,
                              X=X,
                              y=y,
                              safeguard_step=safeguard_step,
                              gd_steps=params
                              )
    else:
        fp_eval_partial = partial(fp_eval_lm_logisticgd,
                                supervised=supervised,
                                z_star=z_star,
                                X=X,
                                y=y,
                                gd_step=params
                                )
    
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    val = z0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def k_steps_train_lm_logisticgd(k, z0, q, params, num_points, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    fp_train_partial = partial(fp_train_lm_logisticgd,
                               supervised=supervised,
                               z_star=z_star,
                               X=X,
                               y=y,
                               gd_step=params
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses

def fp_eval_lm_logisticgd(i, val, supervised, z_star, X, y, gd_step):
    z, loss_vec, z_all, obj_diffs = val
    z_next = fixed_point_logisticgd_lm(z, X, y, gd_step)
    diff = jnp.linalg.norm(z - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    w, b = z[:-1], z[-1]
    obj = compute_loss(y, sigmoid(X @ w + b))
    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y, sigmoid(X @ w_star + b_star))


    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def fp_train_lm_logisticgd(i, val, supervised, z_star, X, y, gd_step):
    z, loss_vec = val
    z_next = fixed_point_logisticgd_lm(z, X, y, gd_step)
    diff = jnp.linalg.norm(z_next - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fixed_point_logisticgd_lm(z, X, y, gd_step):
    w, b = z[:-1], z[-1]
    y_hat = sigmoid(X @ w + b)
    dw, db = compute_gradient(X, y, y_hat)
    w_next = w - gd_step[:-1] * dw #(dw + .01 * w)
    b_next = b - gd_step[-1] * db #(db + .01 * b)
    z_next = jnp.hstack([w_next, b_next])
    return z_next

# def k_steps_eval_nesterov_logisticgd():

def k_steps_eval_nesterov_logisticgd(k, z0, q, params, num_points, supervised, z_star, jit, safeguard=True):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    fp_eval_partial = partial(fp_eval_nesterov_logisticgd,
                            supervised=supervised,
                            z_star=z_star,
                            X=X,
                            y=y,
                            gd_steps=params
                            )
    
    z_all = jnp.zeros((k, z0.size))
    obj_diffs = jnp.zeros(k)
    val = z0, z0, 0, iter_losses, z_all, obj_diffs
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, y_final, t_final, iter_losses, z_all, obj_diffs = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def fp_eval_nesterov_logisticgd(i, val, supervised, z_star, X, y, gd_steps):
    z, v, t, loss_vec, z_all, obj_diffs = val
    z_next, v_next, t_next = fixed_point_nesterov_logisticgd(z, v, t, X, y, gd_steps)
    diff = jnp.linalg.norm(z - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    w, b = z[:-1], z[-1]
    obj = compute_loss(y, sigmoid(X @ w + b))
    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y, sigmoid(X @ w_star + b_star))

    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, v_next, t_next, loss_vec, z_all, obj_diffs


def fixed_point_nesterov_logisticgd(z, v, t, X, y, gd_step):
    v_next = fixed_point_logisticgd(z, X, y, gd_step)
    # t_next = .5 * (1 + jnp.sqrt(1 + 4 * t ** 2))
    z_next = v_next + t/ (t + 3) * (v_next - v)
    t_next = t + 1
    return z_next, v_next, t_next