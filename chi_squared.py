import numpy as np
import scipy.interpolate as itp
import scipy.integrate as integra
from power_spectrum import *
from multipoles import *
from data import Data
from model import Model

def broadband(q, data_poles, model_poles, cov_inv, exp_list):
    # q can be either s or k
    
    V = np.concatenate(data_poles) - np.concatenate(model_poles)
    
    # broad-band terms
    n_exp = len(exp_list)
    n_q = len(q)
    A = np.zeros((2 * n_exp, 2 * n_q))
    for i in range(0, n_exp):
        A[i, : n_q] = q ** exp_list[i]
        A[n_exp + i, n_q:] = q ** exp_list[i]

    # Analytical marginalisation
    alpha_ = A @ cov_inv @ A.T
    beta_ = A @ cov_inv @ V
    coeffs = np.linalg.solve(alpha_, beta_)
    bb = A.T @ coeffs
    
    n_poles = len(data_poles)
    coeffs = np.split(coeffs, n_poles)
    bb = np.split(bb, n_poles)

    return bb, coeffs


def chi2(q, data_poles, model_poles, cov_inv, bb_exp):
    V = np.concatenate(model_poles) - np.concatenate(data_poles)
    bb, coeffs = broadband(q, data_poles, model_poles, cov_inv, bb_exp)

    Vn = V + np.concatenate(bb)
    chi2 = Vn.T @ cov_inv @ Vn

    return chi2


def loglike(q, data_poles, model_poles, cov_inv, bb_exp):
    return -0.5 * chi2(q, data_poles, model_poles, cov_inv, bb_exp)
