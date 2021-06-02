from oracles import eps, lim_val
import numpy as np


phi_val = (1 + np.sqrt(5)) / 2.0


def psi(F, dF, x, L, tau, y):
    """
    Local model \psi_{x, L, \tau}(y) and \hat{\psi}_{x, L, \tau}(y, B) evaluated at point y.
    Parameters
    ----------
    F : array_like
        The system of equations evaluated at point x.
    dF : array_like
        The jacobian of system of equations evaluated at point x.
    x : array_like
        Anchor point for the local model.
    L : float
        The estimate of local Lipschitz constant.
    tau : float
        The hyperparameter of local model.
    y : array_like
        The evaluation point for local model.
    Returns
    -------
    float
        The value of local model evaluated at point y.
    """
    return tau / 2.0 + L * np.sum(np.square(y - x)) / 2.0 +\
        np.sum(np.square(F + np.dot(dF, y - x))) / (2.0 * tau)


def factor_step_probe(F, dF, dF2=None):
    """
    Factor computation of the next point in optimization procedure using spectral decomposition and
    Sherman-Morrison-Woodbury formula.
    Parameters
    ----------
    F : array_like
        The system of equations evaluated at point x.
    dF : array_like
        The jacobian of system of equations evaluated at point x.
    dF2 : array_like, default=None
        If not None, the doubly stochastic step is used and dF2 is tracted as independently
        sampled jacobian.
    Returns
    -------
    Tuple
        The tuple of factors for fast computation of the optimization step:
        Lambda, Q, ... and other factors
        Lambda : array_like
            The diagonal matrix of eigenvalues of hessian-like matrix.
        Q : array_like
            The unitary matrix of eigenvectors for corresponding eigenvalues.
    """
    m, n = dF.shape
    if m > n:
        if dF2 is None:
            Lambda, Q = np.linalg.eigh(np.dot(dF.T, dF))
        else:
            Lambda, Q = np.linalg.eigh(np.dot(dF2.T, dF2))
        return Lambda, Q, np.dot(Q.T, np.dot(dF.T, F))
    if dF2 is None:
        Lambda, Q = np.linalg.eigh(np.dot(dF, dF.T))
        return Lambda, Q, Lambda * np.dot(Q.T, F)
    Lambda, Q = np.linalg.eigh(np.dot(dF2, dF2.T))
    return Lambda, Q, np.dot(dF.T, F), np.dot(dF2.T, Q), np.dot(Q.T, np.dot(dF2, np.dot(dF.T, F)))


def probe_x(x, eta, B, v):
    """
    Computation of the next point in optimization procedure: x - eta * B^{-1}v.
    Parameters
    ----------
    x : array_like
        Current optimizable point in the procedure.
    eta : float
        The step scale.
    B : array_like
        Hessian-like matrix evaluated at x.
    v : array_like
        Gradient of 0.5 * f_2(x) evaluated at x.
    Returns
    -------
    array_like
        The next optimizable point.
    """
    return x - eta * np.dot(np.linalg.inv(np.clip(B, a_min=-lim_val, a_max=lim_val)), v)


def fast_probe_x(x, eta, tauL, F, dF, Lambda, Q, factored_QF, dF2=None):
    """
    Computation of the next point in optimization procedure using spectral decomposition and
    Sherman-Morrison-Woodbury formula.
    Parameters
    ----------
    x : array_like
        Current optimizable point in the procedure.
    eta : float
        The step scale.
    tauL : float
        The value of \tau L.
    F : array_like
        The system of equations evaluated at point x.
    dF : array_like
        The jacobian of system of equations evaluated at point x.
    Lambda : array_like
        The diagonal matrix of eigenvalues of hessian-like matrix.
    Q : array_like
        The unitary matrix of eigenvectors for corresponding eigenvalues.
    factored_QF : tuple
        The tuple of matrices and vectors from factorization of computation of the next point.
    dF2 : array_like, default=None
        If not None, the doubly stochastic step is used and dF2 is tracted as independently
        sampled jacobian.
    Returns
    -------
    array_like
        The next optimizable point.
    """
    m, n = dF.shape
    if m > n:
        return x - eta * np.dot(Q, factored_QF[0] / np.clip(Lambda + tauL, a_min=eps, a_max=lim_val))
    if dF2 is None:
        return x - eta * np.dot(
            dF.T, F - np.dot(Q, factored_QF[0] / np.clip(Lambda + tauL, a_min=eps, a_max=lim_val))) /\
            np.clip(tauL, a_min=eps, a_max=lim_val)
    return x - eta * (
        factored_QF[0] - np.dot(
            factored_QF[1], factored_QF[2] / np.clip(Lambda + tauL, a_min=eps, a_max=lim_val))) /\
            np.clip(tauL, a_min=eps, a_max=lim_val)


def f1_directional_subgradient(oracle, x_new, x_old, t):
    """
    Computes \nabla_t f_1(x_new + t * (x_new - x_old)).
    Parameters
    ----------
    oracle : Oracle class instance
        Oracle of the optimization criterion.
    x_new : array_like
        Current optimizable point in the procedure.
    x_old : array_like
        Previous optimizable point in the procedure.
    t : float
        Momentum.
    Returns
    -------
    subgradient : float
        \nabla_t f_1(x_new + t * (x_new - x_old)) value.
    """
    point = x_new + t * (x_new - x_old)
    f1 = oracle.f_1(point)
    if f1 < eps:
        nabla_f1 = np.random.randn(oracle.shape[0])
        nabla_f1 = nabla_f1 / np.linalg.norm(nabla_f1)
    else:
        nabla_f1 = oracle.F(point) / f1
    return np.dot(np.dot(oracle.dF(point).T, nabla_f1), x_new - x_old)


def golden_ratio_search(oracle, x_new, x_old):
    """
    Auxiliary procedure to compute optimal momentum using the golden-section search.
    Parameters
    ----------
    oracle : Oracle class instance
        Oracle of the optimization criterion.
    x_new : array_like
        Current optimizable point in the procedure.
    x_old : array_like
        Previous optimizable point in the procedure.
    Returns
    -------
    t : float
        The optimal momentum.
    n_iter : int
        The number of iterations per each momentum computation.
    """
    a, b = 0.0, 1.0
    n_iter = 0
    while np.abs(b - a) > eps:
        n_iter += 1
        x1, x2 = b - (b - a) / phi_val, a + (b - a) / phi_val
        y1, y2 = oracle.f_1(x_new + x1 * (x_new - x_old)), oracle.f_1(x_new + x2 * (x_new - x_old))
        if y1 > y2:
            a = x1
        else:
            b = x2
    t = (a + b) / 2.0
    if oracle.f_1(x_new) < oracle.f_1(x_new + t * (x_new - x_old)):
        return 0.0, n_iter
    return t, n_iter


def point_sampling_search(oracle, x_new, x_old, n_points=10, strategy="grid_search"):
    """
    Auxiliary procedure to compute optimal momentum using brute-force search.
    Parameters
    ----------
    oracle : Oracle class instance
        Oracle of the optimization criterion.
    x_new : array_like
        Current optimizable point in the procedure.
    x_old : array_like
        Previous optimizable point in the procedure.
    n_points : int, default=10
        Number of probing points.
    strategy : str, default="grid_search"
        The way of brute-forcing: "grid_search" or "random_search"; "grid_search" stands for uniform lattice of points,
        "random_search" stands for uniform sampling of points.
    Returns
    -------
    t : float
        The optimal momentum.
    """
    if strategy == "grid_search":
        t_vals = np.linspace(0, 1, n_points)
    else:
        t_vals = np.hstack(([0.0], np.random.rand(n_points - 1)))
    min_idx, min_val = 0, oracle.f_1(x_new)
    for idx, t in enumerate(t_vals):
        f1 = oracle.f_1(x_new + t * (x_new - x_old))
        if f1 < min_val:
            min_val = f1
            min_idx = idx
    return t_vals[min_idx]


def armijo_search(oracle, x_new, x_old, c1=0.33, c2=0.66):
    """
    Auxiliary procedure to compute optimal momentum using Armijo rule.
    Parameters
    ----------
    oracle : Oracle class instance
        Oracle of the optimization criterion.
    x_new : array_like
        Current optimizable point in the procedure.
    x_old : array_like
        Previous optimizable point in the procedure.
    c1 : float, default=0.33
        Slope coefficient, 0 < c1 < c2 < 1.
    c2 : float, default=0.66
        Slope coefficient, 0 < c1 < c2 < 1.
    Returns
    -------
    t : float
        The optimal momentum.
    local_steps : int
        The number of localization iterations.
    spec_steps : int
        The number of specialization iterations.
    """
    local_steps, spec_steps = 0, 0
    directional_nabla_f1 = f1_directional_subgradient(oracle, x_new, x_old, 0.0)
    if directional_nabla_f1 >= 0.0:
        return 0.0, local_steps, spec_steps
    f1_zero = oracle.f_1(x_new)
    t = 1.0
    f1_t = oracle.f_1(x_new + t * (x_new - x_old))
    local_steps += 1
    if ((f1_zero + c2 * directional_nabla_f1 * t) <= f1_t) and (f1_t <= (f1_zero + c1 * directional_nabla_f1 * t)):
        return t, local_steps, spec_steps
    t1, t2 = t, t
    while True:
        local_steps += 1
        if ((f1_zero + c2 * directional_nabla_f1 * t1) <= oracle.f_1(x_new + t1 * (x_new - x_old))) and (oracle.f_1(x_new + t1 * (x_new - x_old)) <= (f1_zero + c1 * directional_nabla_f1 * t1)):
            return t1, local_steps, spec_steps
        if ((f1_zero + c2 * directional_nabla_f1 * t2) <= oracle.f_1(x_new + t2 * (x_new - x_old))) and (oracle.f_1(x_new + t2 * (x_new - x_old)) <= (f1_zero + c1 * directional_nabla_f1 * t2)):
            return t2, local_steps, spec_steps
        if ((f1_zero + c2 * directional_nabla_f1 * t1) > oracle.f_1(x_new + t1 * (x_new - x_old))) and (oracle.f_1(x_new + t2 * (x_new - x_old)) <= (f1_zero + c1 * directional_nabla_f1 * t2)):
            t1 = t2
            t2 = 2.0 * t1
        elif ((f1_zero + c2 * directional_nabla_f1 * t1) <= oracle.f_1(x_new + t1 * (x_new - x_old))) and (oracle.f_1(x_new + t2 * (x_new - x_old)) > (f1_zero + c1 * directional_nabla_f1 * t2)):
            t2 = t1
            t1 = t2 / 2.0
        elif ((f1_zero + c2 * directional_nabla_f1 * t1) > oracle.f_1(x_new + t1 * (x_new - x_old))) and (oracle.f_1(x_new + t2 * (x_new - x_old)) > (f1_zero + c1 * directional_nabla_f1 * t2)):
            break
        else:
            return 0.0, local_steps, spec_steps#impossible variant
    while True:
        spec_steps += 1
        hat_t = (t1 + t2) / 2.0
        if ((f1_zero + c2 * directional_nabla_f1 * hat_t) <= oracle.f_1(x_new + hat_t * (x_new - x_old))) and\
            (oracle.f_1(x_new + hat_t * (x_new - x_old)) <= (f1_zero + c1 * directional_nabla_f1 * hat_t)):
            return hat_t, local_steps, spec_steps
        if ((f1_zero + c2 * directional_nabla_f1 * hat_t) > oracle.f_1(x_new + hat_t * (x_new - x_old))) and (oracle.f_1(x_new + t2 * (x_new - x_old)) > (f1_zero + c1 * directional_nabla_f1 * t2)):
            t1 = hat_t
        elif ((f1_zero + c2 * directional_nabla_f1 * t1) > oracle.f_1(x_new + t1 * (x_new - x_old))) and (oracle.f_1(x_new + hat_t * (x_new - x_old)) > (f1_zero + c1 * directional_nabla_f1 * hat_t)):
            t2 = hat_t
        else:
            return 0.0, local_steps, spec_steps#impossible variant


def extrapolation_search(oracle, x_new, x_old):
    """
    Auxiliary procedure to compute optimal momentum using Extrapolation strategy.
    Parameters
    ----------
    oracle : Oracle class instance
        Oracle of the optimization criterion.
    x_new : array_like
        Current optimizable point in the procedure.
    x_old : array_like
        Previous optimizable point in the procedure.
    Returns
    -------
    t : float
        The optimal momentum.
    n_iter : int
        The number of iterations.
    """
    n_iter = 0
    t_old, t_new = 0.0, 1.0
    while (f1_directional_subgradient(oracle, x_new, x_old, t_new) < 0.0) and (oracle.f_1(x_new + t_new * (x_new - x_old)) <= oracle.f_1(x_new + t_old * (x_new - x_old))):
        n_iter += 1
        t_old = t_new
        t_new *= 2.0
    return t_old, n_iter

