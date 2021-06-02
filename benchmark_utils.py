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
    for oracle_class, name in [(NesterovSkokovOracle, 'Nesterov-Skokov'), (HatOracle, 'Hat'), (PLOracle, 'PL')]:
        if args.verbose:
            print('Oracle:', name)
        exp_res_dict['DetGNM'][name] = dict()
        for n in args.n_dims:
            if args.verbose:
                print('    n:', n)
            exp_res_dict['DetGNM'][name][n] = dict()
            for i in range(args.n_starts):
                if args.verbose:
                    print('        start #:', i + 1)
                start = time.time()
                _, f_vals, nabla_f_2_norm_vals, _, _ = DetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, None)
                start = time.time() - start
                exp_res_dict['DetGNM'][name][n][i] = {'f_vals': f_vals, 'nabla_f_2_norm_vals': nabla_f_2_norm_vals, 'avg_time_s': start / len(f_vals), 'time_s': start}
                del _, f_vals, nabla_f_2_norm_vals, start
                gc.collect()
                
    if args.verbose:
        print("Started ArmijoAccDetGNM!")
    exp_res_dict['ArmijoAccDetGNM'] = dict()
    for oracle_class, name in [(NesterovSkokovOracle, 'Nesterov-Skokov'), (HatOracle, 'Hat'), (PLOracle, 'PL')]:
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
                for i in range(args.n_starts):
                    if args.verbose:
                        print('            start #:', i + 1)
                    start = time.time()
                    _, f_vals, nabla_f_2_norm_vals, _, _, local_steps_list, spec_steps_list = AccDetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, None, "Armijo", c1=c1, c2=c2)
                    start = time.time() - start
                    exp_res_dict['ArmijoAccDetGNM'][name][n][pair_num][i] = {'f_vals': f_vals,
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
    for oracle_class, name in [(NesterovSkokovOracle, 'Nesterov-Skokov'), (HatOracle, 'Hat'), (PLOracle, 'PL')]:
        if args.verbose:
            print('Oracle:', name)
        exp_res_dict['ExtrapolationAccDetGNM'][name] = dict()
        for n in args.n_dims:
            if args.verbose:
                print('    n:', n)
            exp_res_dict['ExtrapolationAccDetGNM'][name][n] = dict()
            for i in range(args.n_starts):
                if args.verbose:
                    print('        start #:', i + 1)
                start = time.time()
                _, f_vals, nabla_f_2_norm_vals, _, _, n_iter_list = AccDetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, None, "Extrapolation")
                start = time.time() - start
                exp_res_dict['ExtrapolationAccDetGNM'][name][n][i] = {'f_vals': f_vals,
                                                                      'nabla_f_2_norm_vals': nabla_f_2_norm_vals,
                                                                      'n_iter_list': n_iter_list,
                                                                      'avg_time_s': start / len(f_vals),
                                                                      'time_s': start}
                del _, f_vals, nabla_f_2_norm_vals, n_iter_list, start
                gc.collect()
    
    if args.verbose:
        print("Started InterpolationAccDetGNM!")
    exp_res_dict['InterpolationAccDetGNM'] = dict()
    for oracle_class, name in [(NesterovSkokovOracle, 'Nesterov-Skokov'), (HatOracle, 'Hat'), (PLOracle, 'PL')]:
        if args.verbose:
            print('Oracle:', name)
        exp_res_dict['InterpolationAccDetGNM'][name] = dict()
        for n in args.n_dims:
            if args.verbose:
                print('    n:', n)
            exp_res_dict['InterpolationAccDetGNM'][name][n] = dict()
            for n_points in args.n_points_list:
                if args.verbose:
                    print('        n_points:', n_points)
                exp_res_dict['InterpolationAccDetGNM'][name][n][n_points] = dict()
                for i in range(args.n_starts):
                    if args.verbose:
                        print('            start #:', i + 1)
                    start = time.time()
                    _, f_vals, nabla_f_2_norm_vals, _, _ = AccDetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, None, "Interpolation")
                    start = time.time() - start
                    exp_res_dict['InterpolationAccDetGNM'][name][n][n_points][i] = {'f_vals': f_vals,
                                                                                    'nabla_f_2_norm_vals': nabla_f_2_norm_vals,
                                                                                    'avg_time_s': start / len(f_vals),
                                                                                    'time_s': start}
                    del _, f_vals, nabla_f_2_norm_vals, start
                    gc.collect()
    
    if args.verbose:
        print("Started SamplingAccDetGNM!")
    exp_res_dict['SamplingAccDetGNM'] = dict()
    for oracle_class, name in [(NesterovSkokovOracle, 'Nesterov-Skokov'), (HatOracle, 'Hat'), (PLOracle, 'PL')]:
        if args.verbose:
            print('Oracle:', name)
        exp_res_dict['SamplingAccDetGNM'][name] = dict()
        for n in args.n_dims:
            if args.verbose:
                print('    n:', n)
            exp_res_dict['SamplingAccDetGNM'][name][n] = dict()
            for n_points in args.n_points_list:
                if args.verbose:
                    print('        n_points:', n_points)
                exp_res_dict['SamplingAccDetGNM'][name][n][n_points] = dict()
                for i in range(args.n_starts):
                    if args.verbose:
                        print('            start #:', i + 1)
                    start = time.time()
                    _, f_vals, nabla_f_2_norm_vals, _, _ = AccDetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, None, "Sampling")
                    start = time.time() - start
                    exp_res_dict['SamplingAccDetGNM'][name][n][n_points][i] = {'f_vals': f_vals,
                                                                               'nabla_f_2_norm_vals': nabla_f_2_norm_vals,
                                                                               'avg_time_s': start / len(f_vals),
                                                                               'time_s': start}
                    del _, f_vals, nabla_f_2_norm_vals, start
                    gc.collect()
    
    if args.verbose:
        print("Started GoldenRatioAccDetGNM!")
    exp_res_dict['GoldenRatioAccDetGNM'] = dict()
    for oracle_class, name in [(NesterovSkokovOracle, 'Nesterov-Skokov'), (HatOracle, 'Hat'), (PLOracle, 'PL')]:
        if args.verbose:
            print('Oracle:', name)
        exp_res_dict['GoldenRatioAccDetGNM'][name] = dict()
        for n in args.n_dims:
            if args.verbose:
                print('    n:', n)
            exp_res_dict['GoldenRatioAccDetGNM'][name][n] = dict()
            for i in range(args.n_starts):
                if args.verbose:
                    print('        start #:', i + 1)
                start = time.time()
                _, f_vals, nabla_f_2_norm_vals, _, _, n_iter_list = AccDetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, None, "GoldenRatio")
                start = time.time() - start
                exp_res_dict['GoldenRatioAccDetGNM'][name][n][i] = {'f_vals': f_vals,
                                                                    'nabla_f_2_norm_vals': nabla_f_2_norm_vals,
                                                                    'n_iter_list': n_iter_list,
                                                                    'avg_time_s': start / len(f_vals),
                                                                    'time_s': start}
                del _, f_vals, nabla_f_2_norm_vals, n_iter_list, start
                gc.collect()
    
    return exp_res_dict

