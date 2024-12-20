import sys

import hydra

import lah.examples.jamming as jamming
import lah.examples.lasso as lasso
import lah.examples.markowitz as markowitz
import lah.examples.maxcut as maxcut
import lah.examples.mnist as mnist
import lah.examples.mpc as mpc
import lah.examples.mri as mri
import lah.examples.osc_mass as osc_mass
import lah.examples.phase_retrieval as phase_retrieval
import lah.examples.quadcopter as quadcopter
import lah.examples.robust_kalman as robust_kalman
import lah.examples.robust_ls as robust_ls
import lah.examples.robust_pca as robust_pca
import lah.examples.sine as sine
import lah.examples.sparse_coding as sparse_coding
import lah.examples.sparse_pca as sparse_pca
import lah.examples.unconstrained_qp as unconstrained_qp
import lah.examples.vehicle as vehicle
import lah.examples.ridge_regression as ridge_regression
import lah.examples.logistic_regression as logistic_regression
import lah.examples.opf as opf


@hydra.main(config_path='configs/markowitz', config_name='markowitz_setup.yaml')
def main_setup_markowitz(cfg):
    markowitz.setup_probs(cfg)


@hydra.main(config_path='configs/osc_mass', config_name='osc_mass_setup.yaml')
def main_setup_osc_mass(cfg):
    osc_mass.setup_probs(cfg)


@hydra.main(config_path='configs/vehicle', config_name='vehicle_setup.yaml')
def main_setup_vehicle(cfg):
    vehicle.setup_probs(cfg)


@hydra.main(config_path='configs/quadcopter', config_name='quadcopter_setup.yaml')
def main_setup_quadcopter(cfg):
    quadcopter.setup_probs(cfg)


@hydra.main(config_path='configs/mnist', config_name='mnist_setup.yaml')
def main_setup_mnist(cfg):
    mnist.setup_probs(cfg)


@hydra.main(config_path='configs/jamming', config_name='jamming_setup.yaml')
def main_setup_jamming(cfg):
    jamming.setup_probs(cfg)


@hydra.main(config_path='configs/robust_kalman', config_name='robust_kalman_setup.yaml')
def main_setup_robust_kalman(cfg):
    robust_kalman.setup_probs(cfg)


@hydra.main(config_path='configs/robust_pca', config_name='robust_pca_setup.yaml')
def main_setup_robust_pca(cfg):
    robust_pca.setup_probs(cfg)


@hydra.main(config_path='configs/robust_ls', config_name='robust_ls_setup.yaml')
def main_setup_robust_ls(cfg):
    robust_ls.setup_probs(cfg)


@hydra.main(config_path='configs/sparse_pca', config_name='sparse_pca_setup.yaml')
def main_setup_sparse_pca(cfg):
    sparse_pca.setup_probs(cfg)


@hydra.main(config_path='configs/phase_retrieval', config_name='phase_retrieval_setup.yaml')
def main_setup_phase_retrieval(cfg):
    phase_retrieval.setup_probs(cfg)


@hydra.main(config_path='configs/lasso', config_name='lasso_setup.yaml')
def main_setup_lasso(cfg):
    lasso.setup_probs(cfg)


@hydra.main(config_path='configs/mpc', config_name='mpc_setup.yaml')
def main_setup_mpc(cfg):
    mpc.setup_probs(cfg)


@hydra.main(config_path='configs/unconstrained_qp', config_name='unconstrained_qp_setup.yaml')
def main_setup_unconstrained_qp(cfg):
    unconstrained_qp.setup_probs(cfg)


@hydra.main(config_path='configs/sparse_coding', config_name='sparse_coding_setup.yaml')
def main_setup_sparse_coding(cfg):
    sparse_coding.setup_probs(cfg)


@hydra.main(config_path='configs/sine', config_name='sine_setup.yaml')
def main_setup_sine(cfg):
    sine.setup_probs(cfg)


@hydra.main(config_path='configs/mri', config_name='mri_setup.yaml')
def main_setup_mri(cfg):
    mri.setup_probs(cfg)


@hydra.main(config_path='configs/maxcut', config_name='maxcut_setup.yaml')
def main_setup_maxcut(cfg):
    maxcut.setup_probs(cfg)


@hydra.main(config_path='configs/ridge_regression', config_name='ridge_regression_setup.yaml')
def main_setup_ridge_regression(cfg):
    ridge_regression.setup_probs(cfg)


@hydra.main(config_path='configs/logistic_regression', config_name='logistic_regression_setup.yaml')
def main_setup_logistic_regression(cfg):
    logistic_regression.setup_probs(cfg)


@hydra.main(config_path='configs/opf', config_name='opf_setup.yaml')
def main_setup_opf(cfg):
    opf.setup_probs(cfg)


if __name__ == '__main__':
    if sys.argv[2] == 'cluster':
        base = 'hydra.run.dir=/scratch/gpfs/rajivs/learn2warmstart/outputs/'
    elif sys.argv[2] == 'local':
        base = 'hydra.run.dir=outputs/'
    if sys.argv[1] == 'markowitz':
        # step 1. remove the markowitz argument -- otherwise hydra uses it as an override
        # step 2. add the train_outputs/... argument for data_setup_outputs not outputs
        # sys.argv[1] = 'hydra.run.dir=outputs/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv[1] = base + 'markowitz/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_markowitz()
    elif sys.argv[1] == 'osc_mass':
        sys.argv[1] = base + 'osc_mass/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_osc_mass()
    elif sys.argv[1] == 'vehicle':
        sys.argv[1] = base + 'vehicle/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_vehicle()
    elif sys.argv[1] == 'robust_kalman':
        sys.argv[1] = base + 'robust_kalman/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_robust_kalman()
    elif sys.argv[1] == 'robust_pca':
        sys.argv[1] = base + 'robust_pca/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_robust_pca()
    elif sys.argv[1] == 'robust_ls':
        sys.argv[1] = base + 'robust_ls/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_robust_ls()
    elif sys.argv[1] == 'sparse_pca':
        sys.argv[1] = base + 'sparse_pca/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_sparse_pca()
    elif sys.argv[1] == 'phase_retrieval':
        sys.argv[1] = base + 'phase_retrieval/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_phase_retrieval()
    elif sys.argv[1] == 'lasso':
        sys.argv[1] = base + 'lasso/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_lasso()
    elif sys.argv[1] == 'mpc':
        sys.argv[1] = base + 'mpc/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_mpc()
    elif sys.argv[1] == 'unconstrained_qp':
        sys.argv[1] = base + 'unconstrained_qp/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_unconstrained_qp()
    elif sys.argv[1] == 'quadcopter':
        sys.argv[1] = base + 'quadcopter/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_quadcopter()
    elif sys.argv[1] == 'mnist':
        sys.argv[1] = base + 'mnist/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_mnist()
    elif sys.argv[1] == 'jamming':
        sys.argv[1] = base + 'jamming/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_jamming()
    elif sys.argv[1] == 'sparse_coding':
        sys.argv[1] = base + 'sparse_coding/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_sparse_coding()
    elif sys.argv[1] == 'sine':
        sys.argv[1] = base + 'sine/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_sine()
    elif sys.argv[1] == 'mri':
        sys.argv[1] = base + 'mri/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_mri()
    elif sys.argv[1] == 'maxcut':
        sys.argv[1] = base + 'maxcut/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_maxcut()
    elif sys.argv[1] == 'ridge_regression':
        sys.argv[1] = base + 'ridge_regression/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_ridge_regression()
    elif sys.argv[1] == 'logistic_regression':
        sys.argv[1] = base + 'logistic_regression/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_logistic_regression()
    elif sys.argv[1] == 'opf':
        sys.argv[1] = base + 'opf/data_setup_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_setup_opf()
