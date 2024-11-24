import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import yaml
from lah.launcher import Workspace
from lah.examples.solve_script import gd_setup_script
import os


def run(run_cfg, lah=True):
    example = "ridge_regression"
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
    m_orig = setup_cfg['m_orig']

    lambd = setup_cfg['lambd']
    # A = np.random.normal(size=(m_orig, n_orig)) #/ 10
    D = np.random.normal(size=(m_orig, n_orig)) / np.sqrt(m_orig)
    A = jnp.array(D / np.linalg.norm(D, axis=0))
    # P = A.T @ A + lambd * np.identity(n_orig)
    P = A.T @ A  + lambd * np.identity(n_orig)

    gd_step = 1 / P.max()

    gauss_mean = jnp.zeros(n_orig)
    gauss_var = A.T @ A
    static_dict = dict(P=P, gd_step=gd_step, gauss_mean=gauss_mean, gauss_var=gauss_var)

    # we directly save q now
    static_flag = True
    if lah:
        algo = 'lah_stochastic_gd'
        # algo = 'lah_gd'
    else:
        algo = 'lm_gd'
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example)

    # run the workspace
    workspace.run()


def l2ws_run(run_cfg):
    example = "ridge_regression"
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
    m_orig = setup_cfg['m_orig']

    lambd = setup_cfg['lambd']
    # A = np.random.normal(size=(m_orig, n_orig)) #/ 10
    D = np.random.normal(size=(m_orig, n_orig)) / np.sqrt(m_orig)
    A = jnp.array(D / np.linalg.norm(D, axis=0))
    P = A.T @ A + lambd * np.identity(n_orig)

    evals, evecs = np.linalg.eigh(P)
    mu = evals.min()
    L = evals.max()

    gd_step = 2 / (mu + L)

    static_dict = dict(P=P, gd_step=gd_step)

    # we directly save q now
    static_flag = True
    algo = 'gd'
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example)

    # run the workspace
    workspace.run()


def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    np.random.seed(setup_cfg['seed'])
    n_orig = setup_cfg['n_orig']
    m_orig = setup_cfg['m_orig']

    # b_min = setup_cfg['b_min']
    # b_max = setup_cfg['b_max']
    # rhs_mean = setup_cfg['rhs_mean']
    # rhs_var = setup_cfg['rhs_var']

    lambd = setup_cfg['lambd']
    # A = np.random.normal(size=(m_orig, n_orig))
    D = np.random.normal(size=(m_orig, n_orig)) / np.sqrt(m_orig)
    A = jnp.array(D / np.linalg.norm(D, axis=0))
    P = A.T @ A  + lambd * np.identity(n_orig)

    

    # generate random rhs b vectors
    # b_mat = b_min + (b_max - b_min) * jnp.array(np.random.rand(N, m_orig))
    b_mat = jnp.array(np.random.normal(size=(N, m_orig)))
    theta_mat = b_mat

    c_mat = (-A.T @ b_mat.T).T


    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    gd_setup_script(c_mat, P, theta_mat, output_filename)

    import pdb
    pdb.set_trace()


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
