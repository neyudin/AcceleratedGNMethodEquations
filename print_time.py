import numpy as np


def time_printer(exp_res_dict, args):
    """
    Routine to register time measurements of the whole experiment set.
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
    f = open(args.store_dir + '/' + args.time_stats_fname, 'w')
    print("DetGNM!", file=f)
    for name in ['Rosenbrock-Skokov', 'Hat']:
        print('Oracle:', name, file=f)
        for n in args.n_dims:
            print('    n:', n, file=f)
            for line_search in ["None", "Armijo"]:
                if line_search == "Armijo":
                    print('        line_search:', line_search, file=f)
                    for probe_pair_num, (probe_c1, probe_c2) in enumerate(zip(args.c1_list, args.c2_list)):
                        print('            c1 = {:.4f}, c2 = {:.4f}:'.format(probe_c1, probe_c2), file=f)
                        avg_mean_time, avg_mean_sqr_time, mean_time, mean_sqr_time = 0., 0., 0., 0.
                        mean_iter, mean_sqr_iter = 0, 0.
                        for i in range(args.n_starts):
                            avg_mean_time += exp_res_dict['DetGNM'][name][n][line_search][probe_pair_num][i]["avg_time_s"]
                            avg_mean_sqr_time += exp_res_dict['DetGNM'][name][n][line_search][probe_pair_num][i]["avg_time_s"] ** 2
                            mean_time += exp_res_dict['DetGNM'][name][n][line_search][probe_pair_num][i]["time_s"]
                            mean_sqr_time += exp_res_dict['DetGNM'][name][n][line_search][probe_pair_num][i]["time_s"] ** 2
                            mean_iter += len(exp_res_dict['DetGNM'][name][n][line_search][probe_pair_num][i]["f_vals"])
                            mean_sqr_iter += len(exp_res_dict['DetGNM'][name][n][line_search][probe_pair_num][i]["f_vals"]) ** 2
                        avg_mean_time /= args.n_starts
                        avg_mean_sqr_time /= args.n_starts
                        mean_time /= args.n_starts
                        mean_sqr_time /= args.n_starts
                        mean_iter /= args.n_starts
                        mean_sqr_iter /= args.n_starts
                        avg_std = np.sqrt((avg_mean_sqr_time - avg_mean_time ** 2) * args.n_starts / (args.n_starts - 1))
                        std = np.sqrt((mean_sqr_time - mean_time ** 2) * args.n_starts / (args.n_starts - 1))
                        iter_std = np.sqrt((mean_sqr_iter - mean_iter ** 2) * args.n_starts / (args.n_starts - 1))
                        print('                avg mean time {:.4f} s. +- {:.4f} std s; mean time {:.4f} s. +- {:.4f} std s.; mean number of outer iterations {} +- {:.4f}.'.format(avg_mean_time,
                            avg_std, mean_time, std, mean_iter, iter_std), file=f)
                else:
                    avg_mean_time, avg_mean_sqr_time, mean_time, mean_sqr_time = 0., 0., 0., 0.
                    mean_iter, mean_sqr_iter = 0, 0.
                    print('        line_search:', line_search, file=f)
                    for i in range(args.n_starts):
                        avg_mean_time += exp_res_dict['DetGNM'][name][n][line_search][i]["avg_time_s"]
                        avg_mean_sqr_time += exp_res_dict['DetGNM'][name][n][line_search][i]["avg_time_s"] ** 2
                        mean_time += exp_res_dict['DetGNM'][name][n][line_search][i]["time_s"]
                        mean_sqr_time += exp_res_dict['DetGNM'][name][n][line_search][i]["time_s"] ** 2
                        mean_iter += len(exp_res_dict['DetGNM'][name][n][line_search][i]["f_vals"])
                        mean_sqr_iter += len(exp_res_dict['DetGNM'][name][n][line_search][i]["f_vals"]) ** 2
                    avg_mean_time /= args.n_starts
                    avg_mean_sqr_time /= args.n_starts
                    mean_time /= args.n_starts
                    mean_sqr_time /= args.n_starts
                    mean_iter /= args.n_starts
                    mean_sqr_iter /= args.n_starts
                    avg_std = np.sqrt((avg_mean_sqr_time - avg_mean_time ** 2) * args.n_starts / (args.n_starts - 1))
                    std = np.sqrt((mean_sqr_time - mean_time ** 2) * args.n_starts / (args.n_starts - 1))
                    iter_std = np.sqrt((mean_sqr_iter - mean_iter ** 2) * args.n_starts / (args.n_starts - 1))
                    print('            avg mean time {:.4f} s. +- {:.4f} std s; mean time {:.4f} s. +- {:.4f} std s.; mean number of outer iterations {} +- {:.4f}.'.format(avg_mean_time, avg_std,
                        mean_time, std, mean_iter, iter_std), file=f)
    
    print("ArmijoAccDetGNM!", file=f)
    for name in ['Rosenbrock-Skokov', 'Hat']:
        print('Oracle:', name, file=f)
        for n in args.n_dims:
            print('    n:', n, file=f)
            for pair_num, (c1, c2) in enumerate(zip(args.c1_list, args.c2_list)):
                print('        c1 = {:.4f}, c2 = {:.4f}'.format(c1, c2), file=f)
                for line_search in ["None", "Armijo"]:
                    if line_search == "Armijo":
                        print('            line_search: {}'.format(line_search), file=f)
                        for probe_pair_num, (probe_c1, probe_c2) in enumerate(zip(args.c1_list, args.c2_list)):
                            print('                c1 = {:.4f}, c2 = {:.4f}:'.format(probe_c1, probe_c2), file=f)
                            avg_mean_time, avg_mean_sqr_time, mean_time, mean_sqr_time = 0., 0., 0., 0.
                            mean_iter, mean_sqr_iter = 0, 0.
                            for i in range(args.n_starts):
                                avg_mean_time += exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][probe_pair_num][i]["avg_time_s"]
                                avg_mean_sqr_time += exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][probe_pair_num][i]["avg_time_s"] ** 2
                                mean_time += exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][probe_pair_num][i]["time_s"]
                                mean_sqr_time += exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][probe_pair_num][i]["time_s"] ** 2
                                mean_iter += len(exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][probe_pair_num][i]["f_vals"])
                                mean_sqr_iter += len(exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][probe_pair_num][i]["f_vals"]) ** 2
                            avg_mean_time /= args.n_starts
                            avg_mean_sqr_time /= args.n_starts
                            mean_time /= args.n_starts
                            mean_sqr_time /= args.n_starts
                            mean_iter /= args.n_starts
                            mean_sqr_iter /= args.n_starts
                            avg_std = np.sqrt((avg_mean_sqr_time - avg_mean_time ** 2) * args.n_starts / (args.n_starts - 1))
                            std = np.sqrt((mean_sqr_time - mean_time ** 2) * args.n_starts / (args.n_starts - 1))
                            iter_std = np.sqrt((mean_sqr_iter - mean_iter ** 2) * args.n_starts / (args.n_starts - 1))
                            print('                    ' +\
                                'avg mean time {:.4f} s. +- {:.4f} std s; mean time {:.4f} s. +- {:.4f} std s.; mean number of outer iterations {} +- {:.4f}.'.format(avg_mean_time, avg_std,
                                mean_time, std, mean_iter, iter_std), file=f)
                    else:
                        avg_mean_time, avg_mean_sqr_time, mean_time, mean_sqr_time = 0., 0., 0., 0.
                        mean_iter, mean_sqr_iter = 0, 0.
                        print('            line_search: {}'.format(line_search), file=f)
                        for i in range(args.n_starts):
                            avg_mean_time += exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][i]["avg_time_s"]
                            avg_mean_sqr_time += exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][i]["avg_time_s"] ** 2
                            mean_time += exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][i]["time_s"]
                            mean_sqr_time += exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][i]["time_s"] ** 2
                            mean_iter += len(exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][i]["f_vals"])
                            mean_sqr_iter += len(exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][i]["f_vals"]) ** 2
                        avg_mean_time /= args.n_starts
                        avg_mean_sqr_time /= args.n_starts
                        mean_time /= args.n_starts
                        mean_sqr_time /= args.n_starts
                        mean_iter /= args.n_starts
                        mean_sqr_iter /= args.n_starts
                        avg_std = np.sqrt((avg_mean_sqr_time - avg_mean_time ** 2) * args.n_starts / (args.n_starts - 1))
                        std = np.sqrt((mean_sqr_time - mean_time ** 2) * args.n_starts / (args.n_starts - 1))
                        iter_std = np.sqrt((mean_sqr_iter - mean_iter ** 2) * args.n_starts / (args.n_starts - 1))
                        print('                avg mean time {:.4f} s. +- {:.4f} std s; mean time {:.4f} s. +- {:.4f} std s.; mean number of outer iterations {} +- {:.4f}.'.format(avg_mean_time,
                            avg_std, mean_time, std, mean_iter, iter_std), file=f)
    
    print("Started ExtrapolationAccDetGNM!", file=f)
    for name in ['Rosenbrock-Skokov', 'Hat']:
        print('Oracle:', name, file=f)
        for n in args.n_dims:
            print('    n:', n, file=f)
            for line_search in ["None", "Armijo"]:
                if line_search == "Armijo":
                    print('        line_search:', line_search, file=f)
                    for probe_pair_num, (probe_c1, probe_c2) in enumerate(zip(args.c1_list, args.c2_list)):
                        print('            c1 = {:.4f}, c2 = {:.4f}:'.format(probe_c1, probe_c2), file=f)
                        avg_mean_time, avg_mean_sqr_time, mean_time, mean_sqr_time = 0., 0., 0., 0.
                        mean_iter, mean_sqr_iter = 0, 0.
                        for i in range(args.n_starts):
                            avg_mean_time += exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search][probe_pair_num][i]["avg_time_s"]
                            avg_mean_sqr_time += exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search][probe_pair_num][i]["avg_time_s"] ** 2
                            mean_time += exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search][probe_pair_num][i]["time_s"]
                            mean_sqr_time += exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search][probe_pair_num][i]["time_s"] ** 2
                            mean_iter += len(exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search][probe_pair_num][i]["f_vals"])
                            mean_sqr_iter += len(exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search][probe_pair_num][i]["f_vals"]) ** 2
                        avg_mean_time /= args.n_starts
                        avg_mean_sqr_time /= args.n_starts
                        mean_time /= args.n_starts
                        mean_sqr_time /= args.n_starts
                        mean_iter /= args.n_starts
                        mean_sqr_iter /= args.n_starts
                        avg_std = np.sqrt((avg_mean_sqr_time - avg_mean_time ** 2) * args.n_starts / (args.n_starts - 1))
                        std = np.sqrt((mean_sqr_time - mean_time ** 2) * args.n_starts / (args.n_starts - 1))
                        iter_std = np.sqrt((mean_sqr_iter - mean_iter ** 2) * args.n_starts / (args.n_starts - 1))
                        print('                avg mean time {:.4f} s. +- {:.4f} std s; mean time {:.4f} s. +- {:.4f} std s.; mean number of outer iterations {} +- {:.4f}.'.format(avg_mean_time,
                            avg_std, mean_time, std, mean_iter, iter_std), file=f)
                else:
                    avg_mean_time, avg_mean_sqr_time, mean_time, mean_sqr_time = 0., 0., 0., 0.
                    mean_iter, mean_sqr_iter = 0, 0.
                    print('        line_search:', line_search, file=f)
                    for i in range(args.n_starts):
                        avg_mean_time += exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search][i]["avg_time_s"]
                        avg_mean_sqr_time += exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search][i]["avg_time_s"] ** 2
                        mean_time += exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search][i]["time_s"]
                        mean_sqr_time += exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search][i]["time_s"] ** 2
                        mean_iter += len(exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search][i]["f_vals"])
                        mean_sqr_iter += len(exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search][i]["f_vals"]) ** 2
                    avg_mean_time /= args.n_starts
                    avg_mean_sqr_time /= args.n_starts
                    mean_time /= args.n_starts
                    mean_sqr_time /= args.n_starts
                    mean_iter /= args.n_starts
                    mean_sqr_iter /= args.n_starts
                    avg_std = np.sqrt((avg_mean_sqr_time - avg_mean_time ** 2) * args.n_starts / (args.n_starts - 1))
                    std = np.sqrt((mean_sqr_time - mean_time ** 2) * args.n_starts / (args.n_starts - 1))
                    iter_std = np.sqrt((mean_sqr_iter - mean_iter ** 2) * args.n_starts / (args.n_starts - 1))
                    print('            avg mean time {:.4f} s. +- {:.4f} std s; mean time {:.4f} s. +- {:.4f} std s.; mean number of outer iterations {} +- {:.4f}.'.format(avg_mean_time, avg_std,
                        mean_time, std, mean_iter, iter_std), file=f)
    
    return None

