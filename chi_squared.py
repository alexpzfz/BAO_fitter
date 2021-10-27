import numpy as np
import scipy.interpolate as itp
import scipy.integrate as integra
from ps_template import *
from mulipoles import *

def broadband(data, model, cov_inv, space='config'):
    if space=='config':
        s, xi0, xi2 = data
        xi0m, xi2m = model
        V = np.concatenate((xi0, xi2)) - np.concatenate((xi0m, xi2m))

        # broad band terms
        A = np.zeros((6, 2*len(r)))
        for i in range(0, 3):
            A[i, 0:len(r)] = s**(i - 2)
            A[3+i, len(r):] = s**(i - 2)

    elif space=='fourier':
        k, ps0, ps2 = data
        ps0m, ps2m = model
        V = np.concatenate((ps0, ps2)) - np.concatenate((ps0m, ps2m))

        # broad band terms
        A = np.zeros((10, 2*len(k)))
        for i in range(0, 4):
            A[i, 0:len(k)] = k**(i - 3)
            A[5+i, len(k):] = k**(i - 3)
        A[4, 0:len(k)] = k**2
        A[9, len(k):] = k**2

    # Analytical marginalisation
    alpha_ = A @ cov_inv @ A.T
    beta_ = A @ cov_inv @ V
    coeffs = np.linalg.solve(alpha_, beta_)
    bb0, bb2 = np.split( A.T @ coeffs, 2)

    return bb0, bb2, coeffs


def chi2(theta, data, cov_inv, linear_template, mu , S_r, iso, sigmas=None, space='config'):
    if sigmas==None: # for best-fit
        b, beta, alpha_par, alpha_per, S_par, S_per, S_s = theta
    else: # for mcmc with fixed sigmas
        b, beta, alpha_par, alpha_per = theta
        S_par, S_per, S_s = sigmas

    if space=='config':
        s, xi0, xi2 = data
        k, pk_lin = linear_template

        pt = power_spectrum_template(k, mu, pk_lin, S_par, S_per, S_s, S_r, b, beta, iso)
        xi0m, xi2m = two_point_cf_template(s, k, mu, pt, alpha_par, alpha_per)
        V = np.concatenate((xi0m, xi2m)) - np.concatenate((xi0, xi2))
        bb0, bb2, _ = broadband([s, xi0, xi2], [xi0m, xi2m], cov_inv, space='config')

    elif space=='fourier':
        k_data, ps0, ps2 = data
        k_lin, ps_lin = linear_template

        ps0m, ps2m = ps_multipoles_template(k_data, k_lin, mu, pk_lin, S_par, S_per, S_s, S_r, b, beta, iso, alpha_par, alpha_per)
        V = np.concatenate((ps0m, ps2m)) - np.concatenate((ps0, ps2))
        bb0, bb2, _ = broadband([k_data, ps0, ps2], [ps0m, ps2m], cov_inv, space='fourier')

    Vn = V + np.concatenate((bb0, bb2))
    chi2 = Vn.T @ cov_inv @ Vn

    return chi2

def logprior(theta, sigmas=None):
    if sigmas==None:
        b, beta, alpha_par, alpha_per, S_par, S_per, S_s = theta
    else:
        b, beta, alpha_par, alpha_per = theta
        S_par, S_per, S_s = sigmas
        
    lp = 0. if 0.5 < b < 3 else -np.inf
    lp += 0. if 0 < beta < 2 else -np.inf
    lp += 0. if 0.8 < alpha_par < 1.2 else -np.inf
    lp += 0. if 0.8 < alpha_per < 1.2 else -np.inf
    
    if sigmas == None:
        lp += 0. if 0 < S_par < 12 else -np.inf
        lp += 0. if 0 < S_per < 12 else -np.inf
        lp += 0. if 0 < S_s < 8 else -np.inf
    return lp

def loglike(theta, data, cov_inv, linear_template, S_r, iso, sigmas=None, space='config'):
    return -0.5 * chi2(theta, data, cov_inv, linear_template, S_r, iso, sigmas, space)

def logpost(theta, data, cov_inv, linear_template, S_r, iso, sigmas=None, space='config'):
    '''The natural logarithm of the posterior.'''
    return logprior(theta, sigmas) + loglike(theta, data, cov_inv, linear_template, S_r, iso, sigmas, space)
