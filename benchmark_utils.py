from optimizers import *
import gc
import time


def experiment_runner(args, x_0_dict):
    """
    Runner routine which performs the whole experiment set.
    Parameters
    ----------
    args : populated namespace object from ArgumentParser
        The system of equations evaluated at point x.
    x_0_dict : dict
        The dictionary of initial points x.
    Returns
    -------
    dict
        Aggregated experiment data.
    """
    gc.enable()
    gc.collect()
    
    exp_res_dict = dict()
    
    if args.verbose:
        print("Started DetGNM!")
    exp_res_dict['DetGNM'] = dict()
    for oracle_class, name in [(RosenbrockEvenSumOracle, 'Rosenbrock-Skokov'), (HatOracle, 'Hat')]:
        if args.verbose:
            print('Oracle:', name)
        exp_res_dict['DetGNM'][name] = dict()
        for n in args.n_dims:
            if args.verbose:
                print('    n:', n)
            exp_res_dict['DetGNM'][name][n] = dict()
            for line_search in ["None", "Armijo"]:
                if args.verbose:
                    print('        line_search:', line_search)
                exp_res_dict['DetGNM'][name][n][line_search] = dict()
                if line_search == "Armijo":
                    for pair_num, (c1, c2) in enumerate(zip(args.c1_list, args.c2_list)):
                        if args.verbose:
                            print('            c1 = {:.4f}, c2 = {:.4f}:'.format(c1, c2))
                        exp_res_dict['DetGNM'][name][n][line_search][pair_num] = dict()
                        for i in range(args.n_starts):
                            if args.verbose:
                                print('                start #:', i + 1)
                            start = time.time()
                            _, f_vals, nabla_f_2_norm_vals, _, _ = DetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, None, line_search, probe_c1=c1, probe_c2=c2)
                            start = time.time() - start
                            exp_res_dict['DetGNM'][name][n][line_search][pair_num][i] = {'f_vals': f_vals,
                                                                                         'nabla_f_2_norm_vals': nabla_f_2_norm_vals,
                                                                                         'avg_time_s': start / len(f_vals),
                                                                                         'time_s': start}
                            del _, f_vals, nabla_f_2_norm_vals, start
                            gc.collect()
                else:
                    for i in range(args.n_starts):
                        if args.verbose:
                            print('            start #:', i + 1)
                        start = time.time()
                        _, f_vals, nabla_f_2_norm_vals, _, _ = DetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, None, line_search)
                        start = time.time() - start
                        exp_res_dict['DetGNM'][name][n][line_search][i] = {'f_vals': f_vals, 'nabla_f_2_norm_vals': nabla_f_2_norm_vals, 'avg_time_s': start / len(f_vals), 'time_s': start}
                        del _, f_vals, nabla_f_2_norm_vals, start
                        gc.collect()
                
    if args.verbose:
        print("Started ArmijoAccDetGNM!")
    exp_res_dict['ArmijoAccDetGNM'] = dict()
    for oracle_class, name in [(RosenbrockEvenSumOracle, 'Rosenbrock-Skokov'), (HatOracle, 'Hat')]:
        if args.verbose:
            print('Oracle:', name)
        exp_res_dict['ArmijoAccDetGNM'][name] = dict()
        for n in args.n_dims:
            if args.verbose:
                print('    n:', n)
            exp_res_dict['ArmijoAccDetGNM'][name][n] = dict()
            for pair_num, (c1, c2) in enumerate(zip(args.c1_list, args.c2_list)):
                if args.verbose:
                    print('        c1 = {:.4f}, c2 = {:.4f}:'.format(c1, c2))
                exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num] = dict()
                for line_search in ["None", "Armijo"]:
                    if args.verbose:
                        print('            line_search: {}'.format(line_search))
                    exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search] = dict()
                    if line_search == "Armijo":
                        for probe_pair_num, (probe_c1, probe_c2) in enumerate(zip(args.c1_list, args.c2_list)):
                            if args.verbose:
                                print('                c1 = {:.4f}, c2 = {:.4f}:'.format(probe_c1, probe_c2))
                            exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][probe_pair_num] = dict()
                            for i in range(args.n_starts):
                                if args.verbose:
                                    print('                    start #:', i + 1)
                                start = time.time()
                                _, f_vals, nabla_f_2_norm_vals, _, _, local_steps_list, spec_steps_list = AccDetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, None, "Armijo",
                                                                                                                    line_search, c1=c1, c2=c2, probe_c1=probe_c1, probe_c2=probe_c2)
                                start = time.time() - start
                                exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][probe_pair_num][i] = {'f_vals': f_vals,
                                                                                                                      'nabla_f_2_norm_vals': nabla_f_2_norm_vals,
                                                                                                                      'local_steps_list': local_steps_list,
                                                                                                                      'spec_steps_list': spec_steps_list,
                                                                                                                      'avg_time_s': start / len(f_vals),
                                                                                                                      'time_s': start}
                                del _, f_vals, nabla_f_2_norm_vals, local_steps_list, spec_steps_list, start
                                gc.collect()
                    else:
                        for i in range(args.n_starts):
                            if args.verbose:
                                print('                start #:', i + 1)
                            start = time.time()
                            _, f_vals, nabla_f_2_norm_vals, _, _, local_steps_list, spec_steps_list = AccDetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, None, "Armijo",
                                                                                                                line_search, c1=c1, c2=c2)
                            start = time.time() - start
                            exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][line_search][i] = {'f_vals': f_vals,
                                                                                                  'nabla_f_2_norm_vals': nabla_f_2_norm_vals,
                                                                                                  'local_steps_list': local_steps_list,
                                                                                                  'spec_steps_list': spec_steps_list,
                                                                                                  'avg_time_s': start / len(f_vals),
                                                                                                  'time_s': start}
                            del _, f_vals, nabla_f_2_norm_vals, local_steps_list, spec_steps_list, start
                            gc.collect()
    
    if args.verbose:
        print("Started ExtrapolationAccDetGNM!")
    exp_res_dict['ExtrapolationAccDetGNM'] = dict()
    for oracle_class, name in [(RosenbrockEvenSumOracle, 'Rosenbrock-Skokov'), (HatOracle, 'Hat')]:
        if args.verbose:
            print('Oracle:', name)
        exp_res_dict['ExtrapolationAccDetGNM'][name] = dict()
        for n in args.n_dims:
            if args.verbose:
                print('    n:', n)
            exp_res_dict['ExtrapolationAccDetGNM'][name][n] = dict()
            for line_search in ["None", "Armijo"]:
                if args.verbose:
                    print('        line_search:', line_search)
                exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search] = dict()
                if line_search == "Armijo":
                    for probe_pair_num, (probe_c1, probe_c2) in enumerate(zip(args.c1_list, args.c2_list)):
                        if args.verbose:
                            print('            c1 = {:.4f}, c2 = {:.4f}:'.format(probe_c1, probe_c2))
                        exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search][probe_pair_num] = dict()
                        for i in range(args.n_starts):
                            if args.verbose:
                                print('                start #:', i + 1)
                            start = time.time()
                            _, f_vals, nabla_f_2_norm_vals, _, _, n_iter_list = AccDetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, None, "Extrapolation", line_search,
                                                                                          probe_c1=probe_c1, probe_c2=probe_c2)
                            start = time.time() - start
                            exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search][probe_pair_num][i] = {'f_vals': f_vals,
                                                                                                               'nabla_f_2_norm_vals': nabla_f_2_norm_vals,
                                                                                                               'n_iter_list': n_iter_list,
                                                                                                               'avg_time_s': start / len(f_vals),
                                                                                                               'time_s': start}
                            del _, f_vals, nabla_f_2_norm_vals, n_iter_list, start
                            gc.collect()
                else:
                    for i in range(args.n_starts):
                        if args.verbose:
                            print('            start #:', i + 1)
                        start = time.time()
                        _, f_vals, nabla_f_2_norm_vals, _, _, n_iter_list = AccDetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, None, "Extrapolation", line_search)
                        start = time.time() - start
                        exp_res_dict['ExtrapolationAccDetGNM'][name][n][line_search][i] = {'f_vals': f_vals,
                                                                                           'nabla_f_2_norm_vals': nabla_f_2_norm_vals,
                                                                                           'n_iter_list': n_iter_list,
                                                                                           'avg_time_s': start / len(f_vals),
                                                                                           'time_s': start}
                        del _, f_vals, nabla_f_2_norm_vals, n_iter_list, start
                        gc.collect()
    
    return exp_res_dict

