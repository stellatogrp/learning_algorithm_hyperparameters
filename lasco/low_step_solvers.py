import jax.numpy as jnp


def one_step_gd_solver(z_stars_train, z_currs, gradients):
    return jnp.trace(gradients @ (z_currs - z_stars_train).T) / (jnp.linalg.norm(gradients) ** 2)


def two_step_quad_gd_solver(z_stars_train, z_currs, P):
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

    # step 4
    c_0 = A * C - B ** 2
    c_1 = -A * D + B * C
    c_2 = -C ** 2 + B * D

    # step 5
    beta = (-c_1 + jnp.sqrt(c_1 ** 2 - 4 * c_0 * c_2)) / (2 * c_2)
    alpha = (-c_1 - jnp.sqrt(c_1 ** 2 - 4 * c_0 * c_2)) / (2 * c_2)

    # alpha, beta = two_step_young(P)
    return alpha, beta


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
