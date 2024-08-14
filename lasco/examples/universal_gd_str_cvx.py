import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import yaml
from lasco.launcher import Workspace
from lasco.examples.solve_script import gd_setup_script
import os


def run(run_cfg, lasco=True):
    example = "universal_gd_str_cvx"
    data_yaml_filename = 'data_setup_copied.yaml'

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    # set the seed
    np.random.seed(setup_cfg['seed'])
    n_orig = setup_cfg['n_orig']

    A = np.random.normal(size=(n_orig, n_orig))
    P = None #A.T @ A + 1 * np.identity(n_orig)
    gd_step = n_orig # / P.max()
    static_dict = dict(P=P, gd_step=gd_step)

    # we directly save q now
    static_flag = True
    if lasco:
        algo = 'lasco_gd'
    else:
        algo = 'lm_gd'
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example)

    # run the workspace
    workspace.run()



def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    np.random.seed(setup_cfg['seed'])
    n_orig = setup_cfg['n_orig']
    # m_orig = setup_cfg['m_orig']
    str_cvx = setup_cfg['mu_str_cvx']
    smoothness = setup_cfg['L_smooth']

    b_min = setup_cfg['b_min']
    b_max = setup_cfg['b_max']

    # generate random rhs b vectors
    b_mat = b_min + (b_max - b_min) * jnp.array(np.random.rand(N, n_orig))

    # b_min2 = setup_cfg['b_min2']
    # b_max2 = setup_cfg['b_max2']
    # b_mat = b_min + (b_max2 - b_min) * jnp.array(np.random.rand(N, n_orig))

    # sample different PD matrices P with bounds on eigenvalues
    # evals = str_cvx + (smoothness - str_cvx) * jnp.array(np.random.rand(N, n_orig))
    evals = str_cvx + (smoothness - str_cvx) * jnp.array(np.random.choice(2, size=(N, n_orig)))

    num_fix = 500
    evals = evals.at[:num_fix,:].set(str_cvx)
    evals = evals.at[:num_fix,-1].set(smoothness)
    evals = evals.at[num_fix:2*num_fix,:].set(smoothness)
    evals = evals.at[num_fix:2*num_fix,0].set(str_cvx)
    # import pdb
    # pdb.set_trace()

    P_tensor = jnp.zeros((N, n_orig, n_orig))
    theta_mat = jnp.zeros((N, n_orig ** 2 + n_orig))
    for i in range(N):
        rand_mat = jnp.array(np.random.randn(n_orig, n_orig))
        if i > 1e4:
            Q, R = jnp.linalg.qr(rand_mat)
        else:
            Q = jnp.eye(n_orig)
        P = Q @ jnp.diag(evals[i, :]) @ Q.T
        P_tensor = P_tensor.at[i, :, :].set(P)
    

    for i in range(N):
        P = P_tensor[i, :, :]
        theta_mat = theta_mat.at[i, :n_orig].set(b_mat[i, :])
        theta_mat = theta_mat.at[i, n_orig:].set(jnp.ravel(P))

    import pdb
    pdb.set_trace()

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    gd_setup_script(b_mat, P_tensor, theta_mat, output_filename)


def obj(P, c, z):
    return .5 * z.T @ P @ z  + c @ z

def obj_diff(obj, true_obj):
    return (obj - true_obj)


def solve_many_probs_cvxpy(P, c_mat):
    """
    solves many unconstrained qp problems where each problem has a different b vector
    """
    P_inv = jnp.linalg.inv(P)
    z_stars = -P_inv @ c_mat
    objvals = obj(P, c_mat, z_stars)
    return z_stars, objvals
