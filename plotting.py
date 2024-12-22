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
ArmijoAccDetGNM_perf_grad_func.eps                |   Figure 3   
ArmijoAccDetGNM_perf_val_func.eps                 |   Figure 4   
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
    
    for gnm_type in ['DetGNM', 'ExtrapolationAccDetGNM']:
        sns.set(font_scale=1.3)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 16), sharex=False, sharey=False)
        legend_flag = False
        for col, name in enumerate(['Rosenbrock-Skokov', 'Hat']):
            for row, stat_name in enumerate(['nabla_f_2_norm_vals', 'f_vals']):
                for n in args.n_dims:
                    for ls, line_search in zip(["solid", "dashdot"], ["None", "Armijo"]):
                        if line_search == "Armijo":
                            for probe_pair_num, (probe_c1, probe_c2, marker, c, markevery) in enumerate(zip(args.c1_list, args.c2_list, args.marker_list, args.plot_colors, args.mark_deltas)):
                                data_sums = []
                                data_sizes = []
                                data_sums_of_squares = []
                                for iter_counter in range(args.N_iter):
                                    for i in range(args.n_starts):
                                        if iter_counter < len(exp_res_dict[gnm_type][name][n][line_search][probe_pair_num][i][stat_name]):
                                            if iter_counter >= len(data_sums):
                                                data_sums.append(0.0)
                                                data_sizes.append(0)
                                                data_sums_of_squares.append(0.0)
                                            data_sums[iter_counter] += exp_res_dict[gnm_type][name][n][line_search][probe_pair_num][i][stat_name][iter_counter]
                                            data_sizes[iter_counter] += 1
                                            data_sums_of_squares[iter_counter] +=\
                                                exp_res_dict[gnm_type][name][n][line_search][probe_pair_num][i][stat_name][iter_counter] ** 2
                                data_sizes = np.array(data_sizes)
                                data_means = np.array(data_sums) / data_sizes
                                data_stds = np.sqrt(np.abs(data_sizes * (np.array(data_sums_of_squares) / data_sizes - data_means ** 2) /\
                                                    np.where(data_sizes > 1, data_sizes - 1, 1)))
                                label = r'$n = ${}; поиск $\eta_k$ по Армихо, $c_1 = ${:.1e}, $c_2 = ${:.1e}'.format(n, probe_c1, probe_c2)
                                axes[row, col].plot(np.arange(1, data_means.size + 1), data_means, color=c, marker=marker, markevery=max(1, int(data_means.size * markevery)),
                                                   linewidth=2, ls=ls, label=label, markersize=15)
                        else:
                            data_sums = []
                            data_sizes = []
                            data_sums_of_squares = []
                            for iter_counter in range(args.N_iter):
                                for i in range(args.n_starts):
                                    if iter_counter < len(exp_res_dict[gnm_type][name][n][line_search][i][stat_name]):
                                        if iter_counter >= len(data_sums):
                                            data_sums.append(0.0)
                                            data_sizes.append(0)
                                            data_sums_of_squares.append(0.0)
                                        data_sums[iter_counter] += exp_res_dict[gnm_type][name][n][line_search][i][stat_name][iter_counter]
                                        data_sizes[iter_counter] += 1
                                        data_sums_of_squares[iter_counter] +=\
                                            exp_res_dict[gnm_type][name][n][line_search][i][stat_name][iter_counter] ** 2
                            data_sizes = np.array(data_sizes)
                            data_means = np.array(data_sums) / data_sizes
                            data_stds = np.sqrt(np.abs(data_sizes * (np.array(data_sums_of_squares) / data_sizes - data_means ** 2) /\
                                                np.where(data_sizes > 1, data_sizes - 1, 1)))
                            label = r'$n = ${}; $\eta_k$ постоянный'.format(n)
                            axes[row, col].plot(np.arange(1, data_means.size + 1), data_means, color="b", marker="o", markevery=max(1, int(data_means.size * 0.43)),
                                               linewidth=2, ls=ls, label=label, markersize=15)
                axes[row, col].set_yscale('log')
                if col == 0:
                    if stat_name == 'nabla_f_2_norm_vals':
                        axes[row, col].set_ylabel(r'$\|\nabla f_{2}(x_{k})\|$', fontsize=16)
                    else:
                        axes[row, col].set_ylabel(r'$f_{1}(x_{k})$', fontsize=16)
                if row == 1:
                    axes[row, col].set_xlabel(r'Номер внешней итерации, $k$', fontsize=16)
                if name == "Rosenbrock-Skokov":
                    plot_name = r"F_{RS}"
                else:
                    plot_name = r"F_{H}"
                axes[row, col].set_title(r'${}$'.format(plot_name), fontsize=16)
                axes[row, col].axhline(y=eps, color='r', linestyle='-', linewidth=1)
                if not legend_flag:
                    legend_flag = True
                    handles, labels = axes[row, col].get_legend_handles_labels()
        lg = fig.legend(handles, labels, bbox_to_anchor=(0.97, 0.87), fancybox=True, shadow=True)
        plt.savefig(fname=args.store_dir + '/{}.eps'.format(gnm_type), bbox_extra_artists=(lg,), bbox_inches='tight')
        plt.close(fig)
    
    c1c2_pairs = list(zip(args.c1_list, args.c2_list))
    for stat_type, stat_name in [('grad', 'nabla_f_2_norm_vals'), ('val', 'f_vals')]:
        sns.set(font_scale=1.3)
        fig, axes = plt.subplots(nrows=len(args.n_dims), ncols=2, figsize=(16, 6), sharex=False, sharey=False)
        legend_flag = False
        for col, name in enumerate(['Rosenbrock-Skokov', 'Hat']):
            for row, n in enumerate(args.n_dims):
                for pair_num, c, markersize, delta in zip(np.arange(len(c1c2_pairs)), ['b', 'g', 'r', 'k'], [17, 13, 9, 5], [0.0, 0.01, 0.02, 0.03]):
                    for ls, line_search in zip(["solid", "dashdot"], ["None", "Armijo"]):
                        if line_search == "Armijo":
                            for probe_pair_num, (probe_c1, probe_c2, marker, markevery) in enumerate(zip(args.c1_list, args.c2_list, args.marker_list, args.mark_deltas)):
                                data_sums = []
                                data_sizes = []
                                data_sums_of_squares = []
                                for iter_counter in range(args.N_iter):
                                    for i in range(args.n_starts):
                                        if iter_counter < len(exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][probe_pair_num][i][stat_name]):
                                            if iter_counter >= len(data_sums):
                                                data_sums.append(0.0)
                                                data_sizes.append(0)
                                                data_sums_of_squares.append(0.0)
                                            data_sums[iter_counter] += exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][probe_pair_num][i][stat_name][iter_counter]
                                            data_sizes[iter_counter] += 1
                                            data_sums_of_squares[iter_counter] +=\
                                                exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][probe_pair_num][i][stat_name][iter_counter] ** 2
                                data_sizes = np.array(data_sizes)
                                data_means = np.array(data_sums) / data_sizes
                                data_stds = np.sqrt(np.abs(data_sizes * (np.array(data_sums_of_squares) / data_sizes - data_means ** 2) /\
                                                    np.where(data_sizes > 1, data_sizes - 1, 1)))
                                label = r'$c_1 = ${:.1e}, $c_2 = ${:.1e}; поиск $\eta_k$ по Армихо, $c_1 = ${:.1e}, $c_2 = ${:.1e}'.format(*c1c2_pairs[pair_num], probe_c1, probe_c2)
                                if len(args.n_dims) == 1:
                                    axes[col].plot(np.arange(1, data_means.size + 1), data_means, color=c, marker=marker, markevery=max(1, int(data_means.size * (markevery + delta))),
                                                   linewidth=2, ls=ls, label=label, markersize=markersize)
                                elif (len(args.n_dims) > 1) and (col == 0):
                                    axes[row].plot(np.arange(1, data_means.size + 1), data_means, color=c, marker=marker, markevery=max(1, int(data_means.size * (markevery + delta))),
                                                   linewidth=2, ls=ls, label=label, markersize=markersize)
                                else:
                                    axes[row, col].plot(np.arange(1, data_means.size + 1), data_means, color=c, marker=marker, markevery=max(1, int(data_means.size * (markevery + delta))),
                                                        linewidth=2, ls=ls, label=label, markersize=markersize)
                        else:
                            data_sums = []
                            data_sizes = []
                            data_sums_of_squares = []
                            for iter_counter in range(args.N_iter):
                                for i in range(args.n_starts):
                                    if iter_counter < len(exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][i][stat_name]):
                                        if iter_counter >= len(data_sums):
                                            data_sums.append(0.0)
                                            data_sizes.append(0)
                                            data_sums_of_squares.append(0.0)
                                        data_sums[iter_counter] += exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][i][stat_name][iter_counter]
                                        data_sizes[iter_counter] += 1
                                        data_sums_of_squares[iter_counter] +=\
                                            exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][i][stat_name][iter_counter] ** 2
                            data_sizes = np.array(data_sizes)
                            data_means = np.array(data_sums) / data_sizes
                            data_stds = np.sqrt(np.abs(data_sizes * (np.array(data_sums_of_squares) / data_sizes - data_means ** 2) /\
                                                np.where(data_sizes > 1, data_sizes - 1, 1)))
                            label = r'$c_1 = ${:.1e}, $c_2 = ${:.1e}; $\eta_k$ постоянный'.format(*c1c2_pairs[pair_num])
                            if len(args.n_dims) == 1:
                                axes[col].plot(np.arange(1, data_means.size + 1), data_means, color=c, marker="o", markevery=max(1, int(data_means.size * (0.43 + delta))),
                                               linewidth=2, ls=ls, label=label, markersize=markersize)
                            elif (len(args.n_dims) > 1) and (col == 0):
                                axes[row].plot(np.arange(1, data_means.size + 1), data_means, color=c, marker="o", markevery=max(1, int(data_means.size * (0.43 + delta))),
                                               linewidth=2, ls=ls, label=label, markersize=markersize)
                            else:
                                axes[row, col].plot(np.arange(1, data_means.size + 1), data_means, color=c, marker="o", markevery=max(1, int(data_means.size * (0.43 + delta))),
                                                    linewidth=2, ls=ls, label=label, markersize=markersize)
                if len(args.n_dims) == 1:
                    axes[col].set_yscale('log')
                elif (len(args.n_dims) > 1) and (col == 0):
                    axes[row].set_yscale('log')
                else:
                    axes[row, col].set_yscale('log')
                if col == 0:
                    if stat_type == 'grad':
                        if len(args.n_dims) == 1:
                            axes[col].set_ylabel(r'$\|\nabla f_{2}(x_{k})\|$', fontsize=16)
                        elif (len(args.n_dims) > 1) and (col == 0):
                            axes[row].set_ylabel(r'$\|\nabla f_{2}(x_{k})\|$', fontsize=16)
                        else:
                            axes[row, col].set_ylabel(r'$\|\nabla f_{2}(x_{k})\|$', fontsize=16)
                    else:
                        if len(args.n_dims) == 1:
                            axes[col].set_ylabel(r'$f_{1}(x_{k})$', fontsize=16)
                        elif (len(args.n_dims) > 1) and (col == 0):
                            axes[row].set_ylabel(r'$f_{1}(x_{k})$', fontsize=16)
                        else:
                            axes[row, col].set_ylabel(r'$f_{1}(x_{k})$', fontsize=16)
                if (len(args.n_dims) > 1) and (row == 2):
                    axes[row, col].set_xlabel(r'Номер внешней итерации, $k$', fontsize=16)
                elif (len(args.n_dims) == 1):
                    axes[col].set_xlabel(r'Номер внешней итерации, $k$', fontsize=16)
                if name == "Rosenbrock-Skokov":
                    plot_name = r"F_{RS}"
                else:
                    plot_name = r"F_{H}"
                if len(args.n_dims) == 1:
                    axes[col].set_title(r'${}, n = {}$'.format(plot_name, n), fontsize=16)
                    axes[col].axhline(y=eps, color='r', linestyle='-', linewidth=1)
                elif (len(args.n_dims) > 1) and (col == 0):
                    axes[row].set_title(r'${}, n = {}$'.format(plot_name, n), fontsize=16)
                    axes[row].axhline(y=eps, color='r', linestyle='-', linewidth=1)
                else:
                    axes[row, col].set_title(r'${}, n = {}$'.format(plot_name, n), fontsize=16)
                    axes[row, col].axhline(y=eps, color='r', linestyle='-', linewidth=1)
                if not legend_flag:
                    legend_flag = True
                    if len(args.n_dims) == 1:
                        handles, labels = axes[col].get_legend_handles_labels()
                    elif (len(args.n_dims) > 1) and (col == 0):
                        handles, labels = axes[row].get_legend_handles_labels()
                    else:
                        handles, labels = axes[row, col].get_legend_handles_labels()
        lg = plt.legend(handles, labels, bbox_to_anchor=(0.25, -0.15), fancybox=True, shadow=True, loc='upper center')
        plt.savefig(fname=args.store_dir + '/ArmijoAccDetGNM_perf_{}_func.eps'.format(stat_type), bbox_extra_artists=(lg,), bbox_inches='tight')
        plt.close(fig)
    
    return None

