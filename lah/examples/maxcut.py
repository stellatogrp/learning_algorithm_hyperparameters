import numpy as np
import logging
import yaml
from jax import vmap
import pandas as pd
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp
import os
import scs
import cvxpy as cp
import jax.scipy as jsp
import jax.random as jra
from lah.algo_steps import create_M
from scipy.sparse import csc_matrix
from lah.examples.solve_script import setup_script
from lah.launcher import Workspace
from lah.algo_steps import get_scaled_vec_and_factor
from lah.utils.generic_utils import python_fori_loop, unvec_symm, vec_symm

import networkx as nx


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
    }
)
log = logging.getLogger(__name__)


def run(run_cfg, lah=True):
    example = "maxcut"
    data_yaml_filename = 'data_setup_copied.yaml'

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    ######################### TODO
    # set the seed
    np.random.seed(setup_cfg['seed'])
    n_orig = setup_cfg['n_orig']

    # static_dict = static_canon(n_orig, d_mul, rho_x=rho_x, scale=scale)
    P, A, cones = get_P_A_cones(n_orig)
    static_dict = {'M': create_M(P, A), 'cones_dict': cones}
    

    # we directly save q now
    get_q = None
    static_flag = True
    algo = 'lah_scs' if lah else 'lm_scs'
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example)

    # run the workspace
    workspace.run()


def l2ws_run(run_cfg):
    example = "maxcut"
    data_yaml_filename = 'data_setup_copied.yaml'

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    ######################### TODO
    # set the seed
    np.random.seed(setup_cfg['seed'])
    n_orig = setup_cfg['n_orig']

    # static_dict = static_canon(n_orig, d_mul, rho_x=rho_x, scale=scale)
    P, A, cones = get_P_A_cones(n_orig)
    rho_x = 1
    scale = 1
    m, n = A.shape
    M = create_M(P, A)
    algo_factor, scale_vec = get_scaled_vec_and_factor(M, rho_x, scale, scale, m, n,
                                                       cones['z'])
    static_dict = {'M': create_M(P, A), 'cones_dict': cones, 'algo_factor': algo_factor}
    

    # we directly save q now
    get_q = None
    static_flag = True
    algo = 'scs'
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example)

    # run the workspace
    workspace.run()


def get_P_A_cones(n_orig):
    n_orig_choose_2 = int(n_orig * (n_orig + 1) / 2)

    # form P
    P = np.zeros((n_orig_choose_2, n_orig_choose_2))

    # form A
    A_diag = np.zeros((n_orig, n_orig_choose_2))
    for i in range(n_orig):
        curr = np.zeros((n_orig, n_orig))
        curr[i, i] = 1
        A_diag[i, :] = vec_symm(curr)
    A_identity = -np.identity(n_orig_choose_2)
    A = jnp.vstack([A_diag, A_identity])

    # form the cones
    cones = {'z': n_orig, 's': [n_orig], 'l': 0}
    return P, A, cones

def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    n_orig = cfg.n_orig
    n_orig_choose_2 = int(n_orig * (n_orig + 1) / 2)

    m = n_orig + n_orig_choose_2
    n = n_orig_choose_2
    p = cfg.p

    # sample many erdos renyi
    graphs, laplacian_matrices = generate_erdos_renyi_graphs(N, n_orig, p)

    # create theta_mat
    theta_mat = np.zeros((N, n_orig_choose_2))
    for i in range(N):
        theta_mat[i, :] = -vec_symm(laplacian_matrices[i])

    theta_mat_jax = jnp.array(theta_mat)

    # form P, A, cones
    P, A, cones = get_P_A_cones(n_orig)

    # form q_mat
    b = jnp.concatenate([jnp.ones(n_orig), jnp.zeros(n_orig_choose_2)])
    q_mat = jnp.zeros((N, m + n))
    q_mat = q_mat.at[:, :n].set(theta_mat)
    q_mat = q_mat.at[:, n:].set(b)

    np.random.seed(cfg.seed)
    key = jra.PRNGKey(cfg.seed)

    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"

    P_sparse, A_sparse = csc_matrix(P), csc_matrix(A)
    m, n = A.shape

    # create scs solver object
    #    we can cache the factorization if we do it like this
    b_np, c_np = np.array(q_mat[0, n:]), np.array(q_mat[0, :n])

    data = dict(P=P_sparse, A=A_sparse, b=b_np, c=c_np)
    tol_abs = cfg.solve_acc_abs
    tol_rel = cfg.solve_acc_rel
    max_iters = cfg.get('solve_max_iters', 10000)
    solver = scs.SCS(data, cones, eps_abs=tol_abs, eps_rel=tol_rel, max_iters=max_iters)

    setup_script(q_mat, theta_mat_jax, solver, data, cones, output_filename, solve=cfg.solve)


def generate_erdos_renyi_graphs(num_graphs, num_nodes, p):
    graphs = []
    laplacian_matrices = []
    for _ in range(num_graphs):
        G = nx.erdos_renyi_graph(num_nodes, p)
        graphs.append(G)
        laplacian_matrix = nx.laplacian_matrix(G).toarray()  # Convert to dense array
        laplacian_matrices.append(laplacian_matrix)
    return graphs, laplacian_matrices