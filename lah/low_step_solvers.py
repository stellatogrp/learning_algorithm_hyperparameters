import jax.numpy as jnp


def one_step_gd_solver(z_stars_train, z_currs, gradients):
    return jnp.trace(gradients @ (z_currs - z_stars_train).T) / (jnp.linalg.norm(gradients) ** 2)


def three_step_sol(P, z_bar):
    evals, Q = jnp.linalg.eigh(P)
    # import pdb
    # pdb.set_trace()

    # step 3
    A = jnp.sum(evals * z_bar)
    B = jnp.sum(evals ** 2 * z_bar)
    C = jnp.sum(evals ** 3 * z_bar)
    D = jnp.sum(evals ** 4 * z_bar)
    E = jnp.sum(evals ** 5 * z_bar)
    F = jnp.sum(evals ** 6 * z_bar)

    # step 4 d_0 &= ace - b^2e - ad^2 + 2bcd - c^3, \quad d_1 = c^2d - bd^2 - bce + ade + b^2f -acf \\
    # d_2 &= -cd^2 + c^2e + bde - ae^2 - bcf + adf, \quad  d_3 = d^3 -  2cde + be^2 + c^2f - bdf
    d_0 = A * C*E - B ** 2*E - A*D**2 + 2*B*C*D - C**3
    d_1 = C**2*D - B*D**2 - B*C*E + A*D*E + B**2 * F - A*C*F
    d_2 = -C*D**2 + C**2*E + B*D*E - A*E**2 - B*C*F + A*D*F
    d_3 = D**3 - 2 * C *D*E + B*E**2 + C**2*F - B*D*F

    # step 5
    coefficients = jnp.array([d_3, d_2, d_1, d_0]) #jnp.array([d_0, d_1, d_2, d_3])
    roots = jnp.roots(coefficients, strip_zeros=False)
    alpha, beta, gamma = roots[0].real, roots[1].real, roots[2].real
    return alpha, beta, gamma



def one_step_sol(P, z_bar):
    # step 1
    evals, Q = jnp.linalg.eigh(P)

    # step 3
    A = jnp.sum(evals * z_bar)
    B = jnp.sum(evals ** 2 * z_bar)

    alpha = A / B

    # alpha, beta = two_step_young(P)
    return alpha


def two_step_sol(P, z_bar):
    # step 1
    evals, Q = jnp.linalg.eigh(P)

    # step 3
    A = jnp.sum(evals * z_bar)
    B = jnp.sum(evals ** 2 * z_bar)
    C = jnp.sum(evals ** 3 * z_bar)
    D = jnp.sum(evals ** 4 * z_bar)

    # step 4
    c_0 = A * C - B ** 2
    c_1 = -A * D + B * C
    c_2 = -C ** 2 + B * D

    # step 5
    beta = (-c_1 + jnp.sqrt(c_1 ** 2 - 4 * c_0 * c_2)) / (2 * c_2)
    alpha = (-c_1 - jnp.sqrt(c_1 ** 2 - 4 * c_0 * c_2)) / (2 * c_2)

    # alpha, beta = two_step_young(P)
    return alpha, beta


def two_step_data_quad_gd_solver(z_stars_train, z_currs, P):
    N, n = z_stars_train.shape

    # step 1
    evals, Q = jnp.linalg.eigh(P)

    # # step 2
    z_bar = jnp.zeros(n)
    for i in range(N):
        z_bar += Q.T @ (z_currs[i, :] - z_stars_train[i, :]) / N
    z_bar = z_bar * z_bar
    return two_step_sol(P, z_bar)


def one_step_stochastic_quad_gd_solver(gauss_mean, gauss_variance, step_sizes, P):
    z_bar = stochastic_get_z_bar(gauss_mean, gauss_variance, step_sizes, P)
    return one_step_sol(P, z_bar)


def two_step_stochastic_quad_gd_solver(gauss_mean, gauss_variance, step_sizes, P):
    z_bar = stochastic_get_z_bar(gauss_mean, gauss_variance, step_sizes, P)
    return two_step_sol(P, z_bar)


def three_step_stochastic_quad_gd_solver(gauss_mean, gauss_variance, step_sizes, P):
    z_bar = stochastic_get_z_bar(gauss_mean, gauss_variance, step_sizes, P)
    return three_step_sol(P, z_bar)


def stochastic_get_z_bar(gauss_mean, gauss_variance, params, P):
    # N, n = z_stars_train.shape
    n = P.shape[0]

    # step 1
    evals, Q = jnp.linalg.eigh(P)

    # get the matrix
    # mat = -jnp.eye(n)
    step_sizes = jnp.exp(params[0])
    # for j in range(n):

    # diag_mat = -jnp.ones(n)
    # for i in range(step_sizes.size):
    #     diag_mat = diag_mat * (1 -  step_sizes[i] * evals)
    # diag_mat = diag_mat / evals
    # mat = jnp.diag(diag_mat) @ Q.T

    # Compute the product over (1 - step_size * eval) for all step sizes in a vectorized way
    diag_mat = -jnp.ones_like(evals) * jnp.prod(1 - jnp.outer(step_sizes, evals), axis=0)

    # Divide by evals element-wise
    diag_mat = diag_mat / evals

    # Perform the matrix multiplication
    mat = jnp.diag(diag_mat) @ Q.T


    # import pdb
    # pdb.set_trace()
    # mat = mat @ jnp.diag(1 / evals) @ Q.T


    # step 2
    # def step_2(mat, gauss_mean, gauss_variance):
    # Vectorized computation
    a_gauss_mean = mat @ gauss_mean
    a_gauss_var = jnp.einsum('ij,ik,jk->i', mat, mat, gauss_variance)
    z_bar = a_gauss_mean + a_gauss_var
    return z_bar

    z_bar = jnp.zeros(n)
    for j in range(n):
        a_j = mat[j, :]
        z_bar = z_bar.at[j].set(a_j @ gauss_mean + a_j @ gauss_variance @ a_j)
    return z_bar

    # # step 3
    # A = jnp.sum(evals * z_bar ** 2)
    # B = jnp.sum(evals ** 2 * z_bar ** 2)
    # C = jnp.sum(evals ** 3 * z_bar ** 2)
    # D = jnp.sum(evals ** 4 * z_bar ** 2)

    # # step 4
    # c_0 = A * C - B ** 2
    # c_1 = -A * D + B * C
    # c_2 = -C ** 2 + B * D

    # # step 5
    # beta = (-c_1 + jnp.sqrt(c_1 ** 2 - 4 * c_0 * c_2)) / (2 * c_2)
    # alpha = (-c_1 - jnp.sqrt(c_1 ** 2 - 4 * c_0 * c_2)) / (2 * c_2)

    # # alpha, beta = two_step_young(P)
    # return alpha, beta


def three_step_data_quad_gd_solver(z_stars_train, z_currs, P):
    N, n = z_stars_train.shape

    # step 1
    evals, Q = jnp.linalg.eigh(P)

    # step 2
    z_bar = jnp.zeros(n)
    for i in range(N):
        z_bar += Q.T @ (z_currs[i, :] - z_stars_train[i, :]) / N

    # step 3
    A = jnp.sum(evals * z_bar ** 2)
    B = jnp.sum(evals ** 2 * z_bar ** 2)
    C = jnp.sum(evals ** 3 * z_bar ** 2)
    D = jnp.sum(evals ** 4 * z_bar ** 2)
    E = jnp.sum(evals ** 5 * z_bar ** 2)
    F = jnp.sum(evals ** 6 * z_bar ** 2)

    # step 4 d_0 &= ace - b^2e - ad^2 + 2bcd - c^3, \quad d_1 = c^2d - bd^2 - bce + ade + b^2f -acf \\
    # d_2 &= -cd^2 + c^2e + bde - ae^2 - bcf + adf, \quad  d_3 = d^3 -  2cde + be^2 + c^2f - bdf
    d_0 = A * C*E - B ** 2*E - A*D**2 + 2*B*C*D - C**3
    d_1 = C**2*D - B*D**2 - B*C*E + A*D*E + B**2 * F - A*C*F
    d_2 = -C*D**2 + C**2*E + B*D*E - A*E**2 - B*C*F + A*D*F
    d_3 = D**3 - 2 * C *D*E + B*E**2 + C**2*F - B*D*F

    # step 5
    coefficients = jnp.array([d_3, d_2, d_1, d_0]) #jnp.array([d_0, d_1, d_2, d_3])
    roots = jnp.roots(coefficients, strip_zeros=False)
    alpha, beta, gamma = roots[0].real, roots[1].real, roots[2].real

    # alpha, beta = two_step_young(P)
    return alpha, beta, gamma


def two_step_young(P):
    evals, evecs = jnp.linalg.eigh(P)
    L = evals.max()
    mu = evals.min()
    S = jnp.sqrt(L ** 2 + (L - mu) ** 2)
    alpha = 2  / (mu + S)
    beta = 2 / (2 * L + mu - S)
    return alpha, beta


def one_step_prox_gd_solver(z_stars_train, z_currs, gradients):
    pass


        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # from scipy.optimize import minimize

        # P = self.l2ws_model.P
        # N, n = self.z_stars_train.shape
        
        # I = np.eye(n)  # Identity matrix of the same dimension as P

        # def objective(xy):
        #     x, y = xy
        #     total_sum = 0
        #     for i in range(N):
        #         # Compute (I - xP)(I - yP)z_star_i
        #         transformed_z = np.dot((I - x * P), (I - y * P)).dot(-self.z_stars_train[i, :])
        #         total_sum += np.linalg.norm(transformed_z)**2
        #     return total_sum / N
        
        # # Create a grid of (x, y) values
        # x_values = np.linspace(0.0000, .002, 10)
        # y_values = np.linspace(0.0000, .002, 10)
        # X, Y = np.meshgrid(x_values, y_values)

        # initial_guess = [0.04, 0.04]
        

        # # manually solve

        # # step 1
        # evals, Q = np.linalg.eigh(P)
        # # Q = np.hstack(evecs)
         

        # # step 2
        # z_bar = np.zeros(n)
        # for i in range(N):
        #     z_bar += Q.T @ (-self.z_stars_train[i, :]) / N

        # # step 3
        # A = np.sum(evals * z_bar ** 2) #/ N
        # B = np.sum(evals ** 2 * z_bar ** 2) #/ N
        # C = np.sum(evals ** 3 * z_bar ** 2) #/ N
        # D = np.sum(evals ** 4 * z_bar ** 2) #/ N

        # # step 4
        # c_0 = A * C - B ** 2
        # c_1 = -A * D + B * C
        # c_2 = -C ** 2 + B * D

        # # step 5
        # beta = (-c_1 + np.sqrt(c_1 ** 2 - 4 * c_0 * c_2)) / (2 * c_2)
        # alpha = (-c_1 - np.sqrt(c_1 ** 2 - 4 * c_0 * c_2)) / (2 * c_2)
        # import pdb
        # pdb.set_trace()

        # result = minimize(objective, initial_guess, method='BFGS')

        # # Compute the function value for each (x, y) pair
        # Z = np.array([[objective((x, y)) for x in x_values] for y in y_values])

        # # Plot the landscape
        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(X, Y, Z, cmap='viridis')
        # plt.show()

        # # Plot the contour map and the optimal point
        # plt.figure(figsize=(12, 8))
        # contour = plt.contour(X, Y, Z, levels=30, cmap='viridis')
        # plt.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
        # plt.colorbar(contour)

        # import pdb
        # pdb.set_trace()
