from opt_utils import *
from oracles import RosenbrockEvenSumOracle, HatOracle, lim_val, eps


def DetGNM(oracle, N, x_0, L_0, fast_update=False, tau_const=None, step_scale_search=None, **kwargs):
    """
    Find argminimum of f_1 using the deterministic Gauss-Newton method with exact proximal map and
    \tau_k = \hat{f}_1(x_k).
    Parameters
    ----------
    oracle : Oracle class instance
        Oracle of the optimization criterion.
    N : int
        The number of outer iterations.
    x_0 : array_like
        The initial parameter value.
    L_0 : float
        The initial value of local Lipschitz constant.
    fast_update : bool, default=True
        If true, every step is computed using the factor_step_probe and fast_probe_x functions,
        otherwise only probe_x is used.
    tau_const : float, default=None
        If not None, then the constant value is used for tau equal tau_const.
    Returns
    -------
    x : array_like
        The approximated argminimum.
    f_vals : array_like
        The list of \hat{f}_1(x_k) values at each iteration.
    nabla_f_2_norm_vals : array_like
        The list of \|\nabla\hat{f}_2(x_k)\| values at each iteration.
    nabla_f_2_vals : array_like
        The list of \nabla\hat{f}_2(x_k) values at each iteration.
    n_inner_iters : array_like
        The list of numbers of inner iterations per each outer one.
    """
    f_vals, nabla_f_2_norm_vals, nabla_f_2_vals, n_inner_iters = [], [], [], []
    x = x_0.copy()
    L = L_0
    for i in range(N):
        tau = oracle.f_1(x) if tau_const is None else tau_const
        if tau < eps:
            break
        F = oracle.F(x)
        dF = oracle.dF(x)
        
        if step_scale_search == "Armijo":
            Lambda, Q, *factored_QF = factor_step_probe(F, dF)
            tmp_x = armijo_probe_x(oracle, x, tau * L, F, dF, Lambda, Q, factored_QF, **kwargs)
        else:
            if fast_update:
                Lambda, Q, *factored_QF = factor_step_probe(F, dF)
                tmp_x = fast_probe_x(x, 1.0, tau * L, F, dF, Lambda, Q, factored_QF)
            else:
                dFTdF = np.dot(dF.T, dF)
                v = np.dot(dF.T, F)
                try:
                    tmp_x = probe_x(x, 1.0, dFTdF + tau * L * np.eye(x.size), v)
                except np.linalg.LinAlgError as err:
                    print('Singular matrix encountered: {}!'.format(str(err)))
                    tmp_x = probe_x(x, 1.0, tau * L * np.eye(x.size), v)
        
        n = 1
        while oracle.f_1(tmp_x) > psi(F, dF, x, L, tau, tmp_x):
            L *= 2.0
            
            if step_scale_search == "Armijo":
                Lambda, Q, *factored_QF = factor_step_probe(F, dF)
                tmp_x = armijo_probe_x(oracle, x, tau * L, F, dF, Lambda, Q, factored_QF, **kwargs)
            else:
                if fast_update:
                    tmp_x = fast_probe_x(x, 1.0, tau * L, F, dF, Lambda, Q, factored_QF)
                else:
                    try:
                        tmp_x = probe_x(x, 1.0, dFTdF + tau * L * np.eye(x.size), v)
                    except np.linalg.LinAlgError as err:
                        print('Singular matrix encountered: {}!'.format(str(err)))
                        tmp_x = probe_x(x, 1.0, tau * L * np.eye(x.size), v)
            
            n += 1
        L = max(L / 2.0, L_0)
        x = tmp_x.copy()
        
        f_vals.append(oracle.f_1(x))
        nabla_f_2_vals.append(oracle.nabla_f_2(x))
        nabla_f_2_norm_vals.append(np.linalg.norm(nabla_f_2_vals[-1]))
        n_inner_iters.append(n)
        if nabla_f_2_norm_vals[-1] < eps:
            break
    
    return x, f_vals, nabla_f_2_norm_vals, nabla_f_2_vals, n_inner_iters


def AccDetGNM(oracle, N, x_0, L_0, fast_update=False, tau_const=None, search_strategy="Armijo", step_scale_search=None, **kwargs):
    """
    Find argminimum of f_1 using the accelerated deterministic Gauss-Newton method with exact proximal map and
    \tau_k = \hat{f}_1(x_k).
    Parameters
    ----------
    oracle : Oracle class instance
        Oracle of the optimization criterion.
    N : int
        The number of outer iterations.
    x_0 : array_like
        The initial parameter value.
    L_0 : float
        The initial value of local Lipschitz constant.
    fast_update : bool, default=True
        If true, every step is computed using the factor_step_probe and fast_probe_x functions,
        otherwise only probe_x is used.
    tau_const : float, default=None
        If not None, then the constant value is used for tau equal tau_const.
    search_strategy : str, default="Armijo"
        Auxiliary procedure used to compute momentum. Possible values: "Armijo", "Extrapolation", "Interpolation", "Sampling", "GoldenRatio".
        Can take additional named parameters: "Armijo" has c1 and c2, 0 < c1 < c2 < 1, "Interpolation" and "Sampling" have n_points = 1, 2, 3 ... .
    Returns
    -------
    y : array_like
        The approximated argminimum.
    f_vals : array_like
        The list of \hat{f}_1(x_k) values at each iteration.
    nabla_f_2_norm_vals : array_like
        The list of \|\nabla\hat{f}_2(x_k)\| values at each iteration.
    nabla_f_2_vals : array_like
        The list of \nabla\hat{f}_2(x_k) values at each iteration.
    n_inner_iters : array_like
        The list of numbers of inner iterations per each outer one.
    Additional attributes:
        For "Armijo" strategy
            local_steps_list : array_like
                The list of numbers of localization iterations per each momentum computation.
            spec_steps_list : array_like
                The list of numbers of specialization iterations per each momentum computation.
        For "Extrapolation" and "GoldenRatio" strategies:
            n_iter_list : array_like
                The list of numbers of iterations per each momentum computation.
    """
    f_vals, nabla_f_2_norm_vals, nabla_f_2_vals, n_inner_iters = [], [], [], []
    if search_strategy == "Armijo":
        local_steps_list, spec_steps_list = [], []
    elif search_strategy == "Extrapolation":
        n_iter_list = []
    else: # golden_ratio search
        n_iter_list = []
    x = x_0.copy()
    y = x.copy()
    L = L_0
    for i in range(N):
        tau = oracle.f_1(y) if tau_const is None else tau_const
        if tau < eps:
            break
        F = oracle.F(y)
        dF = oracle.dF(y)
        
        if step_scale_search == "Armijo":
            Lambda, Q, *factored_QF = factor_step_probe(F, dF)
            tmp_x = armijo_probe_x(oracle, y, tau * L, F, dF, Lambda, Q, factored_QF, **kwargs)
        else:
            if fast_update:
                Lambda, Q, *factored_QF = factor_step_probe(F, dF)
                tmp_x = fast_probe_x(y, 1.0, tau * L, F, dF, Lambda, Q, factored_QF)
            else:
                dFTdF = np.dot(dF.T, dF)
                v = np.dot(dF.T, F)
                try:
                    tmp_x = probe_x(y, 1.0, dFTdF + tau * L * np.eye(y.size), v)
                except np.linalg.LinAlgError as err:
                    print('Singular matrix encountered: {}!'.format(str(err)))
                    tmp_x = probe_x(y, 1.0, tau * L * np.eye(y.size), v)
        
        n = 1
        while oracle.f_1(tmp_x) > psi(F, dF, y, L, tau, tmp_x):
            L *= 2.0
            
            if step_scale_search == "Armijo":
                Lambda, Q, *factored_QF = factor_step_probe(F, dF)
                tmp_x = armijo_probe_x(oracle, y, tau * L, F, dF, Lambda, Q, factored_QF, **kwargs)
            else:
                if fast_update:
                    tmp_x = fast_probe_x(y, 1.0, tau * L, F, dF, Lambda, Q, factored_QF)
                else:
                    try:
                        tmp_x = probe_x(y, 1.0, dFTdF + tau * L * np.eye(y.size), v)
                    except np.linalg.LinAlgError as err:
                        print('Singular matrix encountered: {}!'.format(str(err)))
                        tmp_x = probe_x(y, 1.0, tau * L * np.eye(y.size), v)
            
            n += 1
        L = max(L / 2.0, L_0)
        
        if search_strategy == "Armijo":
            t, local_steps, spec_steps = armijo_search(oracle, tmp_x, x, **kwargs) # c1, c2, c1 < c2
            local_steps_list.append(local_steps)
            spec_steps_list.append(spec_steps)
        elif search_strategy == "Extrapolation":
            t, n_iter = extrapolation_search(oracle, tmp_x, x)
            n_iter_list.append(n_iter)
        elif search_strategy == "Interpolation":
            t = point_sampling_search(oracle, tmp_x, x, strategy="grid_search", **kwargs) # n_points
        elif search_strategy == "Sampling":
            t = point_sampling_search(oracle, tmp_x, x, strategy="random_search", **kwargs) # n_points
        else: # golden_ratio search
            t, n_iter = golden_ratio_search(oracle, tmp_x, x)
            n_iter_list.append(n_iter)
        y = tmp_x + t * (tmp_x - x)
        
        x = tmp_x.copy()
        
        f_vals.append(oracle.f_1(y))
        nabla_f_2_vals.append(oracle.nabla_f_2(y))
        nabla_f_2_norm_vals.append(np.linalg.norm(nabla_f_2_vals[-1]))
        n_inner_iters.append(n)
        if nabla_f_2_norm_vals[-1] < eps:
            break
    
    if search_strategy == "Armijo":
        return y, f_vals, nabla_f_2_norm_vals, nabla_f_2_vals, n_inner_iters, local_steps_list, spec_steps_list
    elif search_strategy == "Extrapolation":
        return y, f_vals, nabla_f_2_norm_vals, nabla_f_2_vals, n_inner_iters, n_iter_list
    elif search_strategy == "Interpolation":
        return y, f_vals, nabla_f_2_norm_vals, nabla_f_2_vals, n_inner_iters
    elif search_strategy == "Sampling":
        return y, f_vals, nabla_f_2_norm_vals, nabla_f_2_vals, n_inner_iters
    else: # golden_ratio search
        return y, f_vals, nabla_f_2_norm_vals, nabla_f_2_vals, n_inner_iters, n_iter_list

