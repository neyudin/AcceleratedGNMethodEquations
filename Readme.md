# Description
Code to conduct experiments for the paper [**Regularization and acceleration of Gauss-Newton method**](http://crm-en.ics.org.ru/uploads/crmissues/crm_2024_7/30_yudin.pdf).

## Overview

* *run_experiments.py* — main script to perform experiments;
* *oracles.py* — contains classes for optimization criteria;
* *opt_utils.py* — auxiliary functions for optimizers;
* *optimizers.py* — contains Gauss-Newton optimization algorithms;
* *benchmark_utils.py* — routines for designed experiments;
* *plotting.py* — routines for plotting results;
* *print_time.py* — routines for registering time measurements.

Print help in command line in repository directory to list all hyperparameters of the experiments:
```
    python run_experiments.py -h
```
Run the following command in command line in repository directory to obtain all experiment data in current directory:
```
    python run_experiments.py
```

## Requirements

* [NumPy](https://numpy.org/);
* [Matplotlib](https://matplotlib.org/);
* [Seaborn](https://seaborn.pydata.org/).

## References

<a id="1">[1]</a> Yudin N.E., Gasnikov A.V. Regularization and acceleration of Gauss-Newton method // Computer Research and Modeling, 2024, vol. 16, no. 7, pp. 1829-1840, [doi: https://doi.org/10.20537/2076-7633-2024-16-7-1829-1840](https://doi.org/10.20537/2076-7633-2024-16-7-1829-1840).
