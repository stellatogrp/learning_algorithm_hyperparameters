import sys

import hydra

import lah.examples.lasso as lasso
import lah.examples.maxcut as maxcut
import lah.examples.mnist as mnist
import lah.examples.robust_kalman as robust_kalman
import lah.examples.ridge_regression as ridge_regression
import lah.examples.logistic_regression as logistic_regression
from lah.utils.data_utils import copy_data_file, recover_last_datetime




@hydra.main(config_path='configs/lasso', config_name='lasso_run.yaml')
def main_run_lasso(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'lasso'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    lasso.run(cfg)


@hydra.main(config_path='configs/lasso', config_name='lasso_l2ws_run.yaml')
def main_run_lasso_l2ws(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'lasso'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    lasso.run(cfg, model='l2ws')


@hydra.main(config_path='configs/lasso', config_name='lasso_lm_run.yaml')
def main_run_lasso_lm(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'lasso'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    lasso.run(cfg, model='lm')


@hydra.main(config_path='configs/mnist', config_name='mnist_run.yaml')
def main_run_mnist(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'mnist'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    mnist.run(cfg)


@hydra.main(config_path='configs/mnist', config_name='mnist_l2ws_run.yaml')
def main_run_mnist_l2ws(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'mnist'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    mnist.run(cfg, model='l2ws')


@hydra.main(config_path='configs/mnist', config_name='mnist_lm_run.yaml')
def main_run_mnist_lm(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'mnist'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    mnist.run(cfg, model='lm')


@hydra.main(config_path='configs/robust_kalman', config_name='robust_kalman_run.yaml')
def main_run_robust_kalman(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'robust_kalman'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    robust_kalman.run(cfg)


@hydra.main(config_path='configs/robust_kalman', config_name='robust_kalman_l2ws_run.yaml')
def main_run_robust_kalman_l2ws(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'robust_kalman'
    setup_datetime = cfg.data.datetime
    if setup_datetime == '':
        # get the most recent datetime and update datetimes
        setup_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = setup_datetime
    copy_data_file(example, setup_datetime)
    robust_kalman.l2ws_run(cfg)


@hydra.main(config_path='configs/maxcut', config_name='maxcut_run.yaml')
def main_run_maxcut(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'maxcut'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    maxcut.run(cfg)


@hydra.main(config_path='configs/maxcut', config_name='maxcut_l2ws_run.yaml')
def main_run_maxcut_l2ws(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'maxcut'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    maxcut.l2ws_run(cfg)


@hydra.main(config_path='configs/maxcut', config_name='maxcut_lm_run.yaml')
def main_run_maxcut_lm(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'maxcut'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    maxcut.run(cfg, lasco=False)


@hydra.main(config_path='configs/ridge_regression', config_name='ridge_regression_run.yaml')
def main_run_ridge_regression(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'ridge_regression'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    ridge_regression.run(cfg)


@hydra.main(config_path='configs/ridge_regression', config_name='ridge_regression_l2ws_run.yaml')
def main_run_ridge_regression_l2ws(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'ridge_regression'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    ridge_regression.l2ws_run(cfg)


@hydra.main(config_path='configs/ridge_regression', config_name='ridge_regression_lm_run.yaml')
def main_run_ridge_regression_lm(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'ridge_regression'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    ridge_regression.run(cfg, lasco=False)


@hydra.main(config_path='configs/logistic_regression', config_name='logistic_regression_run.yaml')
def main_run_logistic_regression(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'logistic_regression'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    logistic_regression.run(cfg)


@hydra.main(config_path='configs/logistic_regression', config_name='logistic_regression_l2ws_run.yaml')
def main_run_logistic_regression_l2ws(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'logistic_regression'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    logistic_regression.run(cfg, model='l2ws')


@hydra.main(config_path='configs/logistic_regression', config_name='logistic_regression_lm_run.yaml')
def main_run_logistic_regression_lm(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'logistic_regression'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    logistic_regression.run(cfg, model='lm')


@hydra.main(config_path='configs/robust_kalman', config_name='robust_kalman_lm_run.yaml')
def main_run_robust_kalman_lm(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    example = 'robust_kalman'
    agg_datetime = cfg.data.datetime
    if agg_datetime == '':
        # get the most recent datetime and update datetimes
        agg_datetime = recover_last_datetime(orig_cwd, example, 'data_setup')
        cfg.data.datetime = agg_datetime
    copy_data_file(example, agg_datetime)
    robust_kalman.run(cfg, lasco=False)


if __name__ == '__main__':
    if sys.argv[2] == 'cluster':
        base = 'hydra.run.dir=/scratch/gpfs/rajivs/learn2warmstart/outputs/'
    elif sys.argv[2] == 'local':
        base = 'hydra.run.dir=outputs/'
    if sys.argv[1] == 'markowitz':
        # step 1. remove the markowitz argument -- otherwise hydra uses it as an override
        # step 2. add the train_outputs/... argument for train_outputs not outputs
        # sys.argv[1] = 'hydra.run.dir=outputs/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv[1] = base + 'markowitz/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_markowitz()
    elif sys.argv[1] == 'robust_kalman':
        sys.argv[1] = base + 'robust_kalman/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_robust_kalman()
    elif sys.argv[1] == 'robust_kalman_l2ws':
        sys.argv[1] = base + 'robust_kalman/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_robust_kalman_l2ws()
    elif sys.argv[1] == 'lasso':
        sys.argv[1] = base + 'lasso/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_lasso()
    elif sys.argv[1] == 'lasso_l2ws':
        sys.argv[1] = base + 'lasso/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_lasso_l2ws()
    elif sys.argv[1] == 'lasso_lm':
        sys.argv[1] = base + 'lasso/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_lasso_lm()
    elif sys.argv[1] == 'mnist':
        sys.argv[1] = base + 'mnist/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_mnist()
    elif sys.argv[1] == 'mnist_l2ws':
        sys.argv[1] = base + 'mnist/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_mnist_l2ws()
    elif sys.argv[1] == 'mnist_lm':
        sys.argv[1] = base + 'mnist/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_mnist_lm()
    elif sys.argv[1] == 'maxcut':
        sys.argv[1] = base + 'maxcut/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_maxcut()
    elif sys.argv[1] == 'maxcut_l2ws':
        sys.argv[1] = base + 'maxcut/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_maxcut_l2ws()
    elif sys.argv[1] == 'maxcut_lm':
        sys.argv[1] = base + 'maxcut/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_maxcut_lm()
    elif sys.argv[1] == 'ridge_regression':
        sys.argv[1] = base + 'ridge_regression/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_ridge_regression()
    elif sys.argv[1] == 'ridge_regression_l2ws':
        sys.argv[1] = base + 'ridge_regression/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_ridge_regression_l2ws()
    elif sys.argv[1] == 'ridge_regression_lm':
        sys.argv[1] = base + 'ridge_regression/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_ridge_regression_lm()
    elif sys.argv[1] == 'logistic_regression':
        sys.argv[1] = base + 'logistic_regression/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_logistic_regression()
    elif sys.argv[1] == 'logistic_regression_l2ws':
        sys.argv[1] = base + 'logistic_regression/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_logistic_regression_l2ws()
    elif sys.argv[1] == 'logistic_regression_lm':
        sys.argv[1] = base + 'logistic_regression/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_logistic_regression_lm()
    elif sys.argv[1] == 'robust_kalman_lm':
        sys.argv[1] = base + 'robust_kalman/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_robust_kalman_lm()
