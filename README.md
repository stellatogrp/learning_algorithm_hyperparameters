# learning_algorithm_hyperparameters

This repository is by
[Rajiv Sambharya](https://rajivsambharya.github.io/) and [Bartolomeo Stellato](https://stellato.io/),
and contains the Python source code to reproduce the experiments in our paper
"[Learning Algorithm Hyperparameters for Fast Parametric Convex Optimization](broken)."

## Installation
To install the learn_algo_steps package, run
```
git clone https://github.com/stellatogrp/learn_algo_steps.git
pip install -e ".[dev]"
```

## Overall flow and commands
Reproducing our results for our learning algorithm hyperparameters (LAH) method involves 3 steps:
- data setup: solves all of the training / test / validation instances and saves the parameters and solutions
- train: trains the weights of LAH
- plotting:  plots all the results

Experiments can be run from using the commands below:
```
python benchmarks/lah_setup.py <example> local
python benchmarks/lah_train.py <example> local
python benchmarks/plotter_lah.py <example> local
```

Replace the ```<example> ``` with one of the following to run an experiment.
```
ridge_regression
logistic_regression
lasso
mnist
robust_kalman
maxcut
```


## The details of each command

***
#### ```benchmarks/lah_setup.py```

The first script ```lah_setup.py``` creates all of the problem instances and solves them.
The number of problems that are being solved is set in the setup config file.
That config file also includes other parameters that define the problem instances. 
This only needs to be run once for each example.
Depending on the example, this can take some time because thousands problems are being solved.
After running this script, the results (mainly the parameters and the optimal solutions) are saved a file like
```
outputs/maxcut/data_setup_outputs/2024-10-03/14-54-32/
```

***
#### ```benchmarks/lah_train.py```

The second script ```l2ws_train.py``` does the actual training using the output from the prevous setup command.
In particular, in the config file, it takes a datetime that points to the setup output.
By default, it takes the most recent setup if this pointer is empty.
The train config file holds information about the actual training process.
Run this file for each $H$ value to train for that number of fixed-point steps.
By default we do progressive training $10$ steps at a time.
To replicate our results in the paper, no inputs in the config need to be changed.
Periodically, we evaluate the performance of the learned optimizer over both the training and test sets.
Some that may be of interest are
- ```train_unrolls``` (the number of progressive training steps)
- ```step_varying_num``` (the number of iterations in our step-varying phase, $H$ in our paper - set to $50$ by default)
- ```nn_cfg/lr``` (the learning rate of the meta-optimizer)
- ```nn_cfg/epochs``` (the number of epochs to train on for each progressive training piece)
- ```eval_unrolls``` (the number of evaluation iterations to run)

```
outputs/maxcut/train_outputs/2024-10-04/15-50-10/
```
In this folder there are many metrics that are stored.
We highlight the mains ones here (both the raw data in csv files and the corresponding plots in pdf files).


- Primal and dual residuals over the test problems (there are also the results for the training problems)

    ```outputs/maxcut/train_outputs/2024-10-04/15-50-10/primal_residuals_test.csv```

    ```outputs/maxcut/train_outputs/2024-10-04/15-50-10/dual_residuals_test.csv```

- Fixed-point residuals over the training problems 

    ```outputs/maxcut/train_outputs/2022-06-04/15-14-05/plots/iters_compared_train.csv```
    ```outputs/maxcut/train_outputs/2022-06-04/15-14-05/eval_iters_train.pdf```

- Losses over epochs: for training this holds the average loss (the mean square error), for testing we plot the loss at the current number of steps (which will change due to the progressive training strategy)

    ```outputs/maxcut/train_outputs/2024-10-04/15-50-10/train_test_results.csv```
    ```outputs/maxcut/train_outputs/2024-10-04/15-50-10/losses_over_training.pdf```

- The ```accuracies_test``` folder holds the results that are used for the tables. First, it holds the average number of iterations to reach the desired accuracies in terms of the performance metric (either the suboptimality or maximum of primal and dual residuals depending on if the problem is constrained or no) ($0.1$, $0.01$, $0.001$, and $0.0001$ by default).


The results for the non-data-driven methods (e.g., Nesterov's acceleration, the silver step size rule, the nearest neighbor approach) are all run by default before any training is complete without an additional command.

The results for the learned warm start (L2WS) and learned metric (LM) methods can be obtained with
```
python benchmarks/lah_train.py maxcut_l2ws local
python benchmarks/lah_train.py maxcut_lm local
```

***
#### ```benchmarks/plotter_lah.py```
This file contains the code to plot the results and visualize the step sizes.
In this case, you must update the config file (e.g., ```configs/maxcut/maxcut_plot.yaml```) with the correct datetimes used to train the method.