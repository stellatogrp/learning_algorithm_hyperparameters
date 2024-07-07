from functools import partial

import cvxpy as cp
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, lax, vmap
from jax.tree_util import tree_map

from lasco.utils.generic_utils import python_fori_loop, unvec_symm, vec_symm


def k_steps_eval_lasco_logisticgd(k, z0, q, params, num_points, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    fp_eval_partial = partial(fp_eval_lasco_logisticgd,
                              supervised=supervised,
                              z_star=z_star,
                              X=X,
                              y=y,
                              gd_steps=params
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


def k_steps_train_lasco_logisticgd(k, z0, q, params, num_points, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    X_flat = q[:num_points * 784]
    X = jnp.reshape(X_flat, (num_points, 784))
    y = q[num_points * 784:]

    fp_train_partial = partial(fp_train_lasco_logisticgd,
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


def fp_eval_lasco_logisticgd(i, val, supervised, z_star, X, y, gd_steps):
    z, loss_vec, z_all, obj_diffs = val
    z_next = fixed_point_logisticgd(z, X, y, gd_steps[i])
    diff = jnp.linalg.norm(z - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    w, b = z[:-1], z[-1]
    obj = compute_loss(y, sigmoid(X @ w + b)) #+ .01 * .5 * jnp.linalg.norm(w) ** 2 + .01  * .5 * b ** 2 
    w_star, b_star = z_star[:-1], z_star[-1]
    opt_obj = compute_loss(y, sigmoid(X @ w_star + b_star))
    # obj = .5 * jnp.linalg.norm(A @ z - c) ** 2 + lambd * jnp.linalg.norm(z, ord=1)
    # opt_obj = .5 * jnp.linalg.norm(A @ z_star - c) ** 2 + lambd * jnp.linalg.norm(z_star, ord=1)
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def fp_train_lasco_logisticgd(i, val, supervised, z_star, X, y, gd_steps):
    z, loss_vec = val
    z_next = fixed_point_logisticgd(z, X, y, gd_steps[i])
    diff = jnp.linalg.norm(z_next - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fixed_point_logisticgd(z, X, y, gd_step):
    w, b = z[:-1], z[-1]
    y_hat = sigmoid(X @ w + b)
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
