from functools import partial

import cvxpy as cp
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, lax, vmap
from jax.tree_util import tree_map

from lasco.utils.generic_utils import python_fori_loop, unvec_symm, vec_symm


def k_steps_eval_lm_ista(k, z0, q, params, lambd, A, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_lm_ista,
                              supervised=supervised,
                              z_star=z_star,
                              lambd=lambd,
                              A=A,
                              c=q,
                              ista_step=params
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


def k_steps_train_lm_ista(k, z0, q, params, lambd, A, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_lm_ista,
                               supervised=supervised,
                               z_star=z_star,
                               lambd=lambd,
                               A=A,
                               c=q,
                               ista_step=params
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def fp_eval_lm_ista(i, val, supervised, z_star, lambd, A, c, ista_step):
    z, loss_vec, z_all, obj_diffs = val
    z_next = fixed_point_ista(z, A, c, lambd, ista_step)
    diff = jnp.linalg.norm(z - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    obj = .5 * jnp.linalg.norm(A @ z - c) ** 2 + lambd * jnp.linalg.norm(z, ord=1)
    opt_obj = .5 * jnp.linalg.norm(A @ z_star - c) ** 2 + lambd * jnp.linalg.norm(z_star, ord=1)
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def fp_train_lm_ista(i, val, supervised, z_star, lambd, A, c, ista_step):
    z, loss_vec = val
    z_next = fixed_point_ista(z, A, c, lambd, ista_step)
    diff = jnp.linalg.norm(z_next - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def k_steps_eval_lasco_ista(k, z0, q, params, lambd, A, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_lasco_ista,
                              supervised=supervised,
                              z_star=z_star,
                              lambd=lambd,
                              A=A,
                              c=q,
                              ista_steps=params
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


def k_steps_train_lasco_ista(k, z0, q, params, lambd, A, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_lasco_ista,
                               supervised=supervised,
                               z_star=z_star,
                               lambd=lambd,
                               A=A,
                               c=q,
                               ista_steps=params
                               )
    val = z0, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def fp_eval_lasco_ista(i, val, supervised, z_star, lambd, A, c, ista_steps):
    z, loss_vec, z_all, obj_diffs = val
    z_next = fixed_point_ista(z, A, c, lambd, ista_steps[i])
    diff = jnp.linalg.norm(z - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    obj = .5 * jnp.linalg.norm(A @ z - c) ** 2 + lambd * jnp.linalg.norm(z, ord=1)
    opt_obj = .5 * jnp.linalg.norm(A @ z_star - c) ** 2 + lambd * jnp.linalg.norm(z_star, ord=1)
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all, obj_diffs


def fp_train_lasco_ista(i, val, supervised, z_star, lambd, A, c, ista_steps):
    z, loss_vec = val
    z_next = fixed_point_ista(z, A, c, lambd, ista_steps[i])
    diff = jnp.linalg.norm(z_next - z_star)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec



def fixed_point_ista(z, A, b, lambd, ista_step):
    """
    applies the ista fixed point operator
    """
    return soft_threshold(z + ista_step * A.T.dot(b - A.dot(z)), ista_step * lambd)


def soft_threshold(z, alpha):
    """
    soft-thresholding function for ista
    """
    return jnp.clip(jnp.abs(z) - alpha, a_min=0) * jnp.sign(z)


def k_steps_train_fista(k, z0, q, lambd, A, ista_step, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)

    fp_train_partial = partial(fp_train_fista,
                               supervised=supervised,
                               z_star=z_star,
                               A=A,
                               b=q,
                               lambd=lambd,
                               ista_step=ista_step
                               )
    val = z0, z0, 1, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, y_final, t_final, iter_losses = out
    return z_final, iter_losses


def k_steps_eval_fista(k, z0, q, params, lambd, A, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    obj_diffs = jnp.zeros(k)
    z_all_plus_1 = jnp.zeros((k + 1, z0.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z0)
    fp_eval_partial = partial(fp_eval_fista,
                              supervised=supervised,
                              z_star=z_star,
                              A=A,
                              b=q,
                              lambd=lambd,
                              ista_step=params
                              )
    z_all = jnp.zeros((k, z0.size))
    val = z0, z0, 1, iter_losses, obj_diffs, z_all
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, y_final, t_final, iter_losses, obj_diffs, z_all = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, obj_diffs


def fixed_point_fista(z, y, t, A, b, lambd, ista_step):
    """
    applies the fista fixed point operator
    """
    z_next = fixed_point_ista(y, A, b, lambd, ista_step)
    t_next = .5 * (1 + jnp.sqrt(1 + 4 * t ** 2))
    y_next = z_next + (t - 1) / t_next * (z_next - z)
    return z_next, y_next, t_next


def fp_train_fista(i, val, supervised, z_star, A, b, lambd, ista_step):
    z, y, t, loss_vec, obj_diffs = val
    z_next, y_next, t_next = fixed_point_fista(z, y, t, A, b, lambd, ista_step)
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    # diff = eval_ista_obj(z_next, A, b, lambd)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, y_next, t_next, loss_vec, obj_diffs


def fp_eval_fista(i, val, supervised, z_star, A, b, lambd, ista_step):
    z, y, t, loss_vec, obj_diffs, z_all = val
    z_next, y_next, t_next = fixed_point_fista(z, y, t, A, b, lambd, ista_step[i])
    if supervised:
        diff = 10 * jnp.log10(jnp.linalg.norm(z - z_star) ** 2 / jnp.linalg.norm(z_star) ** 2)
        # diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)

    obj = .5 * jnp.linalg.norm(A @ z - b) ** 2 + lambd * jnp.linalg.norm(z, ord=1)
    opt_obj = .5 * jnp.linalg.norm(A @ z_star - b) ** 2 + lambd * jnp.linalg.norm(z_star, ord=1)
    obj_diffs = obj_diffs.at[i].set(obj - opt_obj)

    z_all = z_all.at[i, :].set(z_next)
    return z_next, y_next, t_next, loss_vec, obj_diffs, z_all