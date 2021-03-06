# Description
Code to conduct experiments for the paper **Regularization and acceleration of Gauss-Newton method**.

## Overview

* *run_experiments.py* — main script to perform experiments;
* *oracles.py* — contains classes for optimization criteria;
* *opt_utils.py* — auxiliary functions for optimizers;
* *optimizers.py* — contains Gauss-Newton optimization algorithms;
* *benchmark_utils.py* — routines for designed experiments;
* *plotting.py* — routines for plotting results;
* *print_time.py* — routines for registering time measurements;

Print help to list all hyperparameters of the experiments:
```
    python run_experiments.py -h
```
Run the following command to obtain all experiment data in current directory:
```
    python run_experiments.py
```
