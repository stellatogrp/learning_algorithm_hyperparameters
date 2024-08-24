import jax.numpy as jnp


def one_step_gd_solver(z_stars_train, z_currs, gradients):
    return jnp.sum(z_stars_train @ z_currs) / (jnp.linalg.norm(gradients) ** 2)


def two_step_quad_gd_solver(z_stars_train, z_currs, P):
    N, n = z_stars_train.shape

    # step 1
    evals, Q = jnp.linalg.eigh(P)

    # step 2
    z_bar = jnp.zeros(n)
    for i in range(N):
        z_bar += Q.T @ (z_currs - z_stars_train[i, :]) / N

    # step 3
    A = jnp.sum(evals * z_bar ** 2)
    B = jnp.sum(evals ** 2 * z_bar ** 2)
    C = jnp.sum(evals ** 3 * z_bar ** 2)
    D = jnp.sum(evals ** 4 * z_bar ** 2)

    # step 4
    c_0 = A * C - B ** 2
    c_1 = -A * D + B * C
    c_2 = -C ** 2 + B * D

    # step 5
    beta = (-c_1 + jnp.sqrt(c_1 ** 2 - 4 * c_0 * c_2)) / (2 * c_2)
    alpha = (-c_1 - jnp.sqrt(c_1 ** 2 - 4 * c_0 * c_2)) / (2 * c_2)
    return alpha, beta


def one_step_prox_gd_solver(z_stars_train, z_currs, gradients):
    pass
