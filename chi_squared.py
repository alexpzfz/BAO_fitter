import numpy as np
import scipy.interpolate as itp
import scipy.integrate as integra
from ps_template import *
from multipoles import *

def broadband(data, model, cov_inv, exp_list):
    q, f0, f2 = data  # q can be either s or k, while f can be either xi or ps
    f0m, f2m = model
    V = np.concatenate((f0, f2)) - np.concatenate((f0m, f2m))

    # broad-band terms
    nexp = len(exp_list)
    nq = len(q)
    A = np.zeros((2*nexp, 2*nq))
    for i in range(0, nexp):
        A[i, :nq] = q**exp_list[i]
        A[nexp+i, nq:] = q**exp_list[i]

    # Analytical marginalisation
    alpha_ = A @ cov_inv @ A.T
    beta_ = A @ cov_inv @ V
    coeffs = np.linalg.solve(alpha_, beta_)
    bb0, bb2 = np.split( A.T @ coeffs, 2)

    return bb0, bb2, coeffs


def chi2(theta, data, cov_inv, linear_template, mu , S_r, iso, bb_exp, space, sigmas=None):
    if sigmas==None: # Varying sigmas
        b, beta, alpha_par, alpha_per, S_par, S_per, S_s = theta
    else: # Fixed sigmas
        b, beta, alpha_par, alpha_per = theta
        S_par, S_per, S_s = sigmas

    if space=='config':
        s, xi0, xi2 = data
        k, pk_lin = linear_template

        pt = power_spectrum_template(k, mu, pk_lin, S_par, S_per, S_s, S_r, b, beta, iso)
        xi0m, xi2m = two_point_cf_template(s, k, mu, pt, alpha_par, alpha_per)
        V = np.concatenate((xi0m, xi2m)) - np.concatenate((xi0, xi2))
        bb0, bb2, _ = broadband([s, xi0, xi2], [xi0m, xi2m], cov_inv, bb_exp)

    elif space=='fourier':
        k_data, ps0, ps2 = data
        k_lin, ps_lin = linear_template

        ps0m, ps2m = ps_multipoles_template(k_data, k_lin, mu, ps_lin, S_par,
                                            S_per, S_s, S_r, b, beta, iso, alpha_par, alpha_per)
        V = np.concatenate((ps0m, ps2m)) - np.concatenate((ps0, ps2))
        bb0, bb2, _ = broadband([k_data, ps0, ps2], [ps0m, ps2m], cov_inv, bb_exp)

    Vn = V + np.concatenate((bb0, bb2))
    chi2 = Vn.T @ cov_inv @ Vn

    return chi2


def loglike(theta, data, cov_inv, linear_template, mu, S_r, iso, bb_exp, space, sigmas=None):
    return -0.5 * chi2(theta, data, cov_inv, linear_template, mu, S_r, iso, bb_exp, space, sigmas)
