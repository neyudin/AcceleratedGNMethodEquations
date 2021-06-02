import numpy as np
import matplotlib.pyplot as plt
import warnings
from oracles import eps


warnings.filterwarnings('ignore', category=FutureWarning)


import seaborn as sns


"""
-----------------------------------------------------------------
Mapping between filename and figure from the supplement document 
-----------------------------------------------------------------
                    filename                      | fugure name  
-----------------------------------------------------------------
DetGNM.eps                                        |   Figure 1   
ExtrapolationAccDetGNM.eps                        |   Figure 2   
GoldenRatioAccDetGNM.eps                          |   Figure 3   
ArmijoAccDetGNM_perf_grad_func.eps                |   Figure 4   
ArmijoAccDetGNM_perf_val_func.eps                 |   Figure 5   
InterpolationAccDetGNM_perf_grad_func.eps         |   Figure 6   
InterpolationAccDetGNM_perf_val_func.eps          |   Figure 7   
SamplingAccDetGNM_perf_grad_func.eps              |   Figure 8   
SamplingAccDetGNM_perf_val_func.eps               |   Figure 9   
-----------------------------------------------------------------
"""


def plot_experiments_results(exp_res_dict, args):
    """
    Plotting routine which draws results of the whole experiment set.
    Parameters
    ----------
    exp_res_dict : dict
        The whole infographics of the experiments.
    args : populated namespace object from ArgumentParser
        The system of equations evaluated at point x.
    Returns
    -------
    None
    """
    
    for gnm_type in ['DetGNM', 'ExtrapolationAccDetGNM', 'GoldenRatioAccDetGNM']:
        sns.set(font_scale=1.3)
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 16), sharex=True, sharey=False)
        legend_flag = False
        for col, name in enumerate(['Nesterov-Skokov', 'Hat', 'PL']):
            for row, stat_name in enumerate(['nabla_f_2_norm_vals', 'f_vals']):
                for n, c, marker in zip(args.n_dims, ['b', 'g', 'r'], ['o', '^', 'v']):
                    data_sums = []
                    data_sizes = []
                    data_sums_of_squares = []
                    for iter_counter in range(args.N_iter):
                        for i in range(args.n_starts):
                            if iter_counter < len(exp_res_dict[gnm_type][name][n][i][stat_name]):
                                if iter_counter >= len(data_sums):
                                    data_sums.append(0.0)
                                    data_sizes.append(0)
                                    data_sums_of_squares.append(0.0)
                                data_sums[iter_counter] += exp_res_dict[gnm_type][name][n][i][stat_name][iter_counter]
                                data_sizes[iter_counter] += 1
                                data_sums_of_squares[iter_counter] +=\
                                    exp_res_dict[gnm_type][name][n][i][stat_name][iter_counter] ** 2
                    data_sizes = np.array(data_sizes)
                    data_means = np.array(data_sums) / data_sizes
                    data_stds = np.sqrt(np.abs(data_sizes * (np.array(data_sums_of_squares) / data_sizes - data_means ** 2) /\
                                        np.where(data_sizes > 1, data_sizes - 1, 1)))
                    label = r'$n = ${}'.format(n)
                    axes[row, col].plot(np.arange(1, data_means.size + 1), data_means, color=c, marker=marker, markevery=5,
                                        linewidth=1, ls='--', label=label)
                    #axes[row].fill_between(np.arange(1, data_means.size + 1), data_means - data_stds, data_means + data_stds,
                    #                       facecolor=c, edgecolor=c, linewidth=2, ls='-', antialiased=True)#, alpha=0.35)
                axes[row, col].set_yscale('log')
                if stat_name == 'nabla_f_2_norm_vals':
                    axes[row, col].set_ylabel(r'$\|\nabla f_{2}(x_{k})\|$', fontsize=16)
                else:
                    axes[row, col].set_ylabel(r'$f_{1}(x_{k})$', fontsize=16)
                if row == 1:
                    axes[row, col].set_xlabel(r'Номер внешней итерации, $k$', fontsize=16)
                axes[row, col].set_title(r'${}$'.format(name), fontsize=16)
                axes[row, col].axhline(y=eps, color='r', linestyle='-', linewidth=1)
                if not legend_flag:
                    legend_flag = True
                    handles, labels = axes[row, col].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.9))
        plt.savefig(fname=args.store_dir + '/{}.eps'.format(gnm_type))
        plt.close(fig)
    
    c1c2_pairs = list(zip(args.c1_list, args.c2_list))
    for stat_type, stat_name in [('grad', 'nabla_f_2_norm_vals'), ('val', 'f_vals')]:
        sns.set(font_scale=1.3)
        fig, axes = plt.subplots(nrows=len(args.n_dims), ncols=3, figsize=(20, 18), sharex=True, sharey=False)
        legend_flag = False
        for col, name in enumerate(['Nesterov-Skokov', 'Hat', 'PL']):
            for row, n in enumerate(args.n_dims):
                for pair_num, c, marker in zip(np.arange(len(c1c2_pairs)), ['b', 'g', 'r', 'k'], ['o', '^', 'v', 's']):
                    data_sums = []
                    data_sizes = []
                    data_sums_of_squares = []
                    for iter_counter in range(args.N_iter):
                        for i in range(args.n_starts):
                            if iter_counter < len(exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][i][stat_name]):
                                if iter_counter >= len(data_sums):
                                    data_sums.append(0.0)
                                    data_sizes.append(0)
                                    data_sums_of_squares.append(0.0)
                                data_sums[iter_counter] += exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][i][stat_name][iter_counter]
                                data_sizes[iter_counter] += 1
                                data_sums_of_squares[iter_counter] +=\
                                    exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][i][stat_name][iter_counter] ** 2
                    data_sizes = np.array(data_sizes)
                    data_means = np.array(data_sums) / data_sizes
                    data_stds = np.sqrt(np.abs(data_sizes * (np.array(data_sums_of_squares) / data_sizes - data_means ** 2) /\
                                        np.where(data_sizes > 1, data_sizes - 1, 1)))
                    label = r'$c_1 = ${:.4f}, $c_2 = ${:.4f}'.format(*c1c2_pairs[pair_num])
                    axes[row, col].plot(np.arange(1, data_means.size + 1), data_means, color=c, marker=marker, markevery=5,
                                        linewidth=1, ls='--', label=label)
                    #axes[row].fill_between(np.arange(1, data_means.size + 1), data_means - data_stds, data_means + data_stds,
                    #                       facecolor=c, edgecolor=c, linewidth=2, ls='-', antialiased=True)#, alpha=0.35)
                axes[row, col].set_yscale('log')
                if stat_type == 'grad':
                    axes[row, col].set_ylabel(r'$\|\nabla f_{2}(x_{k})\|$', fontsize=16)
                else:
                    axes[row, col].set_ylabel(r'$f_{1}(x_{k})$', fontsize=16)
                if row == 2:
                    axes[row, col].set_xlabel(r'Номер внешней итерации, $k$', fontsize=16)
                axes[row, col].set_title(r'${}, n = {}$'.format(name, n), fontsize=16)
                axes[row, col].axhline(y=eps, color='r', linestyle='-', linewidth=1)
                if not legend_flag:
                    legend_flag = True
                    handles, labels = axes[row, col].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.9))
        plt.savefig(fname=args.store_dir + '/ArmijoAccDetGNM_perf_{}_func.eps'.format(stat_type))
        plt.close(fig)
    
    for stat_type, stat_name in [('grad', 'nabla_f_2_norm_vals'), ('val', 'f_vals')]:
        sns.set(font_scale=1.3)
        fig, axes = plt.subplots(nrows=len(args.n_dims), ncols=3, figsize=(20, 18), sharex=True, sharey=False)
        legend_flag = False
        for col, name in enumerate(['Nesterov-Skokov', 'Hat', 'PL']):
            for row, n in enumerate(args.n_dims):
                for p, c, marker in zip(args.n_points_list, ['b', 'g', 'r', 'c', 'm', 'y', 'k'], ['o', '^', 'v', 's', '>', 'x', 'd']):
                    data_sums = []
                    data_sizes = []
                    data_sums_of_squares = []
                    for iter_counter in range(args.N_iter):
                        for i in range(args.n_starts):
                            if iter_counter < len(exp_res_dict['InterpolationAccDetGNM'][name][n][p][i][stat_name]):
                                if iter_counter >= len(data_sums):
                                    data_sums.append(0.0)
                                    data_sizes.append(0)
                                    data_sums_of_squares.append(0.0)
                                data_sums[iter_counter] += exp_res_dict['InterpolationAccDetGNM'][name][n][p][i][stat_name][iter_counter]
                                data_sizes[iter_counter] += 1
                                data_sums_of_squares[iter_counter] +=\
                                    exp_res_dict['InterpolationAccDetGNM'][name][n][p][i][stat_name][iter_counter] ** 2
                    data_sizes = np.array(data_sizes)
                    data_means = np.array(data_sums) / data_sizes
                    data_stds = np.sqrt(np.abs(data_sizes * (np.array(data_sums_of_squares) / data_sizes - data_means ** 2) /\
                                        np.where(data_sizes > 1, data_sizes - 1, 1)))
                    label = r'$p = ${}'.format(p)
                    axes[row, col].plot(np.arange(1, data_means.size + 1), data_means, color=c, marker=marker, markevery=5,
                                        linewidth=1, ls='--', label=label)
                    #axes[row].fill_between(np.arange(1, data_means.size + 1), data_means - data_stds, data_means + data_stds,
                    #                       facecolor=c, edgecolor=c, linewidth=2, ls='-', antialiased=True)#, alpha=0.35)
                axes[row, col].set_yscale('log')
                if stat_type == 'grad':
                    axes[row, col].set_ylabel(r'$\|\nabla f_{2}(x_{k})\|$', fontsize=16)
                else:
                    axes[row, col].set_ylabel(r'$f_{1}(x_{k})$', fontsize=16)
                if row == 2:
                    axes[row, col].set_xlabel(r'Номер внешней итерации, $k$', fontsize=16)
                axes[row, col].set_title(r'${}, n = {}$'.format(name, n), fontsize=16)
                axes[row, col].axhline(y=eps, color='r', linestyle='-', linewidth=1)
                if not legend_flag:
                    legend_flag = True
                    handles, labels = axes[row, col].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.9))
        plt.savefig(fname=args.store_dir + '/InterpolationAccDetGNM_perf_{}_func.eps'.format(stat_type))
        plt.close(fig)
    
    for stat_type, stat_name in [('grad', 'nabla_f_2_norm_vals'), ('val', 'f_vals')]:
        sns.set(font_scale=1.3)
        fig, axes = plt.subplots(nrows=len(args.n_dims), ncols=3, figsize=(20, 18), sharex=True, sharey=False)
        legend_flag = False
        for col, name in enumerate(['Nesterov-Skokov', 'Hat', 'PL']):
            for row, n in enumerate(args.n_dims):
                for p, c, marker in zip(args.n_points_list, ['b', 'g', 'r', 'c', 'm', 'y', 'k'], ['o', '^', 'v', 's', '>', 'x', 'd']):
                    data_sums = []
                    data_sizes = []
                    data_sums_of_squares = []
                    for iter_counter in range(args.N_iter):
                        for i in range(args.n_starts):
                            if iter_counter < len(exp_res_dict['SamplingAccDetGNM'][name][n][p][i][stat_name]):
                                if iter_counter >= len(data_sums):
                                    data_sums.append(0.0)
                                    data_sizes.append(0)
                                    data_sums_of_squares.append(0.0)
                                data_sums[iter_counter] += exp_res_dict['SamplingAccDetGNM'][name][n][p][i][stat_name][iter_counter]
                                data_sizes[iter_counter] += 1
                                data_sums_of_squares[iter_counter] +=\
                                    exp_res_dict['SamplingAccDetGNM'][name][n][p][i][stat_name][iter_counter] ** 2
                    data_sizes = np.array(data_sizes)
                    data_means = np.array(data_sums) / data_sizes
                    data_stds = np.sqrt(np.abs(data_sizes * (np.array(data_sums_of_squares) / data_sizes - data_means ** 2) /\
                                        np.where(data_sizes > 1, data_sizes - 1, 1)))
                    label = r'$p = ${}'.format(p)
                    axes[row, col].plot(np.arange(1, data_means.size + 1), data_means, color=c, marker=marker, markevery=5,
                                        linewidth=1, ls='--', label=label)
                    #axes[row].fill_between(np.arange(1, data_means.size + 1), data_means - data_stds, data_means + data_stds,
                    #                       facecolor=c, edgecolor=c, linewidth=2, ls='-', antialiased=True)#, alpha=0.35)
                axes[row, col].set_yscale('log')
                if stat_type == 'grad':
                    axes[row, col].set_ylabel(r'$\|\nabla f_{2}(x_{k})\|$', fontsize=16)
                else:
                    axes[row, col].set_ylabel(r'$f_{1}(x_{k})$', fontsize=16)
                if row == 2:
                    axes[row, col].set_xlabel(r'Номер внешней итерации, $k$', fontsize=16)
                axes[row, col].set_title(r'${}, n = {}$'.format(name, n), fontsize=16)
                axes[row, col].axhline(y=eps, color='r', linestyle='-', linewidth=1)
                if not legend_flag:
                    legend_flag = True
                    handles, labels = axes[row, col].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.9))
        plt.savefig(fname=args.store_dir + '/SamplingAccDetGNM_perf_{}_func.eps'.format(stat_type))
        plt.close(fig)

