from functools import partial

import cvxpy as cp
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, lax, vmap
from jax.tree_util import tree_map
from lah.algo_steps import lin_sys_solve

from lah.utils.generic_utils import python_fori_loop, unvec_symm, vec_symm


def k_steps_eval_lm_osqp(k, z0, q, params, P, A, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    m, n = A.shape

    # initialize z_init
    z_init = jnp.zeros((n + 2 * m))
    z_init = z_init.at[:m + n].set(z0)
    w = A @ z0[:n]
    z_init = z_init.at[m + n:].set(w)

    z_all_plus_1 = jnp.zeros((k + 1, z_init.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z_init)

    # unwrap the params
    factor, scale_vec = params
    n = A.shape[1]
    rho_vec = scale_vec[n:]
    sigma_vec = scale_vec[:n]

    fp_eval_partial = partial(fp_eval_lm_osqp,
                              supervised=supervised,
                              z_star=z_star,
                              factor=factor,
                              P=P,
                              A=A,
                              q=q,
                              rho_vec=rho_vec,
                              sigma_vec=sigma_vec
                              )
    z_all = jnp.zeros((k, z_init.size))
    primal_residuals = jnp.zeros(k)
    dual_residuals = jnp.zeros(k)
    val = z_init, iter_losses, z_all, primal_residuals, dual_residuals
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, primal_residuals, dual_residuals = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, primal_residuals, dual_residuals


def k_steps_train_lm_osqp(k, z0, q, params, P, A, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    m, n = A.shape

    # initialize z_init
    z_init = jnp.zeros((n + 2 * m))
    z_init = z_init.at[:m + n].set(z0)
    w = A @ z0[:n]
    z_init = z_init.at[m + n:].set(w)

    # factor, rho_vec, sigma_vec = params

    # unwrap the params
    factor, scale_vec = params
    n = A.shape[1]
    rho_vec = scale_vec[n:]
    sigma_vec = scale_vec[:n]

    fp_train_partial = partial(fp_train_lm_osqp,
                               supervised=supervised,
                               z_star=z_star,
                               A=A,
                               q=q,
                               factor=factor,
                               rho_vec=rho_vec,
                               sigma_vec=sigma_vec
                               )
    val = z_init, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def fp_eval_lm_osqp(i, val, supervised, z_star, factor, P, A, q, rho_vec, sigma_vec,
                       custom_loss=None, lightweight=False):
    m, n = A.shape
    z, loss_vec, z_all, primal_residuals, dual_residuals = val

    factor1, factor2 = factor
    z_next = fixed_point_osqp(z, factor1, factor2, A, q, rho_vec, sigma_vec)
    if custom_loss is None:
        if supervised:
            diff = jnp.linalg.norm(z - z_star)
        else:
            diff = jnp.linalg.norm(z_next - z)
    else:
        diff = custom_loss(z, z_star)
    loss_vec = loss_vec.at[i].set(diff)
    z_all = z_all.at[i, :].set(z_next)

    # primal and dual residuals
    if not lightweight:
        pr = jnp.linalg.norm(A @ z[:n] - z[n + m:])
        dr = jnp.linalg.norm(P @ z[:n] + A.T @ z[n:n + m] + q[:n])
        # pr = jnp.linalg.norm(A @ z_next[:n] - z_next[n + m:])
        # dr = jnp.linalg.norm(P @ z_next[:n] + A.T @ z_next[n:n + m] + q[:n])
        primal_residuals = primal_residuals.at[i].set(pr)
        dual_residuals = dual_residuals.at[i].set(dr)
    return z_next, loss_vec, z_all, primal_residuals, dual_residuals


def fp_train_lm_osqp(i, val, supervised, z_star, factor, A, q, rho_vec, sigma_vec):
    z, loss_vec = val

    factor1, factor2 = factor

    z_next = fixed_point_osqp(z, factor1, factor2, A, q, rho_vec, sigma_vec)
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


# def fp_eval_lm_ista(i, val, supervised, z_star, lambd, A, c, ista_step):
#     z, loss_vec, z_all, obj_diffs = val
#     z_next = fixed_point_osqp(z, A, c, lambd, ista_step)
#     diff = jnp.linalg.norm(z - z_star)
#     loss_vec = loss_vec.at[i].set(diff)
#     obj = .5 * jnp.linalg.norm(A @ z - c) ** 2 + lambd * jnp.linalg.norm(z, ord=1)
#     opt_obj = .5 * jnp.linalg.norm(A @ z_star - c) ** 2 + lambd * jnp.linalg.norm(z_star, ord=1)
#     obj_diffs = obj_diffs.at[i].set(obj - opt_obj)
#     z_all = z_all.at[i, :].set(z_next)
#     return z_next, loss_vec, z_all, obj_diffs


# def fp_train_lm_osqp(i, val, supervised, z_star, lambd, A, c, ista_step):
#     z, loss_vec = val
#     z_next = fixed_point_osqp(z, A, c, lambd, ista_step)
#     diff = jnp.linalg.norm(z_next - z_star)
#     loss_vec = loss_vec.at[i].set(diff)
#     return z_next, loss_vec


def k_steps_train_osqp(k, z0, q, factor, A, rho, sigma, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    m, n = A.shape

    # initialize z_init
    z_init = jnp.zeros((n + 2 * m))
    z_init = z_init.at[:m + n].set(z0)
    w = A @ z0[:n]
    z_init = z_init.at[m + n:].set(w)

    fp_train_partial = partial(fp_train_osqp,
                               supervised=supervised,
                               z_star=z_star,
                               factor=factor,
                               A=A,
                               q=q,
                               rho=rho,
                               sigma=sigma
                               )
    val = z_init, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def k_steps_eval_osqp(k, z0, q, factor, P, A, rho, sigma, supervised, z_star, jit, custom_loss=None):
    iter_losses = jnp.zeros(k)
    m, n = A.shape

    # initialize z_init
    z_init = jnp.zeros((n + 2 * m))
    z_init = z_init.at[:m + n].set(z0)
    w = A @ z0[:n]
    z_init = z_init.at[m + n:].set(w)

    z_all_plus_1 = jnp.zeros((k + 1, z_init.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z_init)
    fp_eval_partial = partial(fp_eval_osqp,
                              supervised=supervised,
                              z_star=z_star,
                              factor=factor,
                              P=P,
                              A=A,
                              q=q,
                              rho=rho,
                              sigma=sigma,
                              custom_loss=custom_loss
                              )
    z_all = jnp.zeros((k, z_init.size))
    primal_resids, dual_resids = jnp.zeros(k), jnp.zeros(k)
    val = z_init, iter_losses, z_all, primal_resids, dual_resids
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, primal_resids, dual_resids = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, primal_resids, dual_resids


def fp_train_osqp(i, val, supervised, z_star, factor, A, q, rho, sigma):
    z, loss_vec = val
    z_next = fixed_point_osqp_old(z, factor, A, q, rho, sigma)
    if supervised:
        diff = jnp.linalg.norm(z - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fp_eval_osqp(i, val, supervised, z_star, factor, P, A, q, rho, sigma, custom_loss=None, lightweight=False):
    m, n = A.shape
    z, loss_vec, z_all, primal_residuals, dual_residuals = val
    z_next = fixed_point_osqp_old(z, factor, A, q, rho, sigma)
    if custom_loss is None:
        if supervised:
            diff = jnp.linalg.norm(z - z_star)
        else:
            diff = jnp.linalg.norm(z_next - z)
    else:
        diff = custom_loss(z, z_star)
    loss_vec = loss_vec.at[i].set(diff)
    z_all = z_all.at[i, :].set(z_next)

    # primal and dual residuals
    if not lightweight:
        pr = jnp.linalg.norm(A @ z_next[:n] - z_next[n + m:])
        dr = jnp.linalg.norm(P @ z_next[:n] + A.T @ z_next[n:n + m] + q[:n])
        primal_residuals = primal_residuals.at[i].set(pr)
        dual_residuals = dual_residuals.at[i].set(dr)
    return z_next, loss_vec, z_all, primal_residuals, dual_residuals


def fixed_point_osqp(z, factor1, factor2, A, q, rho, sigma):
    # z = (x, y, w) w is the z variable in osqp terminology
    m, n = A.shape
    x, y, w = z[:n], z[n:n + m], z[n + m:]
    c, l_bound, u_bound = q[:n], q[n:n + m], q[n + m:]


    # update (x, nu)
    rhs = sigma * x - c + A.T @ (rho * w - y)
    factor = (factor1, factor2)

    x_next = lin_sys_solve(factor, rhs)
    nu = rho * (A @ x_next - w) + y

    # update w_tilde
    w_tilde = w + (nu - y) / rho

    # update w
    w_next = jnp.clip(w_tilde + y / rho, a_min=l_bound, a_max=u_bound)

    # update y
    y_next = y + rho * (w_tilde - w_next)

    # concatenate into the fixed point vector
    z_next = jnp.concatenate([x_next, y_next, w_next])

    return z_next


def fixed_point_osqp_old(z, factor, A, q, rho, sigma):
    # z = (x, y, w) w is the z variable in osqp terminology
    m, n = A.shape
    x, y, w = z[:n], z[n:n + m], z[n + m:]
    c, l_bound, u_bound = q[:n], q[n:n + m], q[n + m:]

    # update (x, nu)
    rhs = sigma * x - c + A.T @ (rho * w - y)
    x_next = lin_sys_solve(factor, rhs)
    nu = rho * (A @ x_next - w) + y

    # update w_tilde
    w_tilde = w + (nu - y) / rho

    # update w
    w_next = jnp.clip(w_tilde + y / rho, a_min=l_bound, a_max=u_bound)

    # update y
    y_next = y + rho * (w_tilde - w_next)

    # concatenate into the fixed point vector
    z_next = jnp.concatenate([x_next, y_next, w_next])

    return z_next


def k_steps_eval_lah_osqp(k, z0, q, params, P, A, idx_mapping, supervised, z_star, jit, custom_loss=None):
    iter_losses = jnp.zeros(k)
    m, n = A.shape

    # initialize z_init
    z_init = jnp.zeros((n + 2 * m))
    z_init = z_init.at[:m + n].set(z0)
    w = A @ z0[:n]
    z_init = z_init.at[m + n:].set(w)

    scalar_params, all_factors = params[0], params[1]
    rhos, sigmas, alphas = jnp.exp(scalar_params[:, 0]), jnp.exp(scalar_params[:, 1]), scalar_params[:, 2]

    z_all_plus_1 = jnp.zeros((k + 1, z_init.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z_init)
    fp_eval_partial = partial(fp_eval_lah_osqp,
                              supervised=supervised,
                              z_star=z_star,
                              all_factors=all_factors,
                              P=P,
                              A=A,
                              idx_mapping=idx_mapping,
                              q=q,
                              rhos=rhos,
                              sigmas=sigmas,
                              alphas=alphas,
                              custom_loss=custom_loss
                              )
    z_all = jnp.zeros((k, z_init.size))
    primal_resids, dual_resids = jnp.zeros(k), jnp.zeros(k)
    val = z_init, iter_losses, z_all, primal_resids, dual_resids
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all, primal_resids, dual_resids = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1, primal_resids, dual_resids


def k_steps_train_lah_osqp(k, z0, q, params, A, idx_mapping, supervised, z_star, jit):
    iter_losses = jnp.zeros(k)
    m, n = A.shape

    # initialize z_init
    z_init = jnp.zeros((n + 2 * m))
    z_init = z_init.at[:m + n].set(z0)
    w = A @ z0[:n]
    z_init = z_init.at[m + n:].set(w)

    scalar_params, all_factors = params[0], params[1]
    rhos, sigmas, alphas = jnp.exp(scalar_params[:, 0]), jnp.exp(scalar_params[:, 1]), scalar_params[:, 2]
    
    fp_train_partial = partial(fp_train_lah_osqp,
                               supervised=supervised,
                               z_star=z_star,
                               all_factors=all_factors,
                               A=A,
                               idx_mapping=idx_mapping,
                               q=q,
                               rhos=rhos,
                               sigmas=sigmas,
                               alphas=alphas
                               )
    val = z_init, iter_losses
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_train_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_train_partial, val)
    z_final, iter_losses = out
    return z_final, iter_losses


def fp_train_lah_osqp(i, val, supervised, z_star, all_factors, A, idx_mapping, q, rhos, sigmas, alphas):
    z, loss_vec = val

    factors1, factors2 = all_factors
    idx = idx_mapping[i]
    z_next = fixed_point_osqp(z, factors1[idx, :, :], factors2[idx, :], A, q, rhos[idx], sigmas[idx])
    if supervised:
        diff = jnp.linalg.norm(z_next - z_star)
    else:
        diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    return z_next, loss_vec


def fp_eval_lah_osqp(i, val, supervised, z_star, all_factors, P, A, idx_mapping, q, rhos, sigmas, alphas,
                       custom_loss=None, lightweight=False):
    m, n = A.shape
    z, loss_vec, z_all, primal_residuals, dual_residuals = val
    idx = idx_mapping[i]

    factors1, factors2 = all_factors
    z_next = fixed_point_osqp(z, factors1[idx, :, :], factors2[idx, :], A, q, rhos[idx], sigmas[idx])
    if custom_loss is None:
        if supervised:
            diff = jnp.linalg.norm(z - z_star)
        else:
            diff = jnp.linalg.norm(z_next - z)
    else:
        diff = custom_loss(z, z_star)
    loss_vec = loss_vec.at[i].set(diff)
    z_all = z_all.at[i, :].set(z_next)

    # primal and dual residuals
    if not lightweight:
        pr = jnp.linalg.norm(A @ z[:n] - z[n + m:])
        dr = jnp.linalg.norm(P @ z[:n] + A.T @ z[n:n + m] + q[:n])
        # pr = jnp.linalg.norm(A @ z_next[:n] - z_next[n + m:])
        # dr = jnp.linalg.norm(P @ z_next[:n] + A.T @ z_next[n:n + m] + q[:n])
        primal_residuals = primal_residuals.at[i].set(pr)
        dual_residuals = dual_residuals.at[i].set(dr)
    return z_next, loss_vec, z_all, primal_residuals, dual_residuals

