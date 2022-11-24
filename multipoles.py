import numpy as np
import scipy.interpolate as itp
import scipy.integrate as integrate
from power_spectrum import *

# Legendre polyonomials
def lp(x, ell):
    if ell == 0:
        l = 1.
    elif ell == 2:
        l = 0.5 * (3 * x ** 2 - 1)
    elif ell == 4:
        l = 0.125 * (35 * x ** 4 - 30 * x ** 2 + 3)
    else:
        raise Exception("This function only accepts ell=0,2,4.")
    return l

# Spherical Bessel functions
def jb(x, ell):
    if ell == 0:
        j = np.sin(x) / x
    elif ell == 2:
        j = (3 / (x ** 2) - 1) * (np.sin(x) / x) - 3 * np.cos(x) / (x ** 2)
    elif ell == 4:
        j = (1 / (x ** 5)) * (5 * x * (2 * x ** 2 - 21) * np.cos(x) + (x ** 4 - 45 * x ** 2 + 105) * np.sin(x))
    else:
        raise Exception("This function only accepts ell=0,2,4.")
    return j


# Power spectrum multipoles
def pk_ell(p2d, mu, ell):
    integrand = p2d * lp(mu, ell)
    prefactor = 2 * ell + 1
    pkell = prefactor * integrate.simps(integrand, mu)

    return pkell

# Configuration space mutlipoles
def xi_ell(s, ell, pk_ell, k):
    # hankel transform to configuration space multipoles, with Gaussian filter
    prefactor = (0.5 / ((np.pi) ** 2)) * (k ** 3) * np.exp(-k * 2)
    if ell % 4 != 0:
        prefactor *= -1.
    integrand = prefactor * pk_ell * jb(s[:, None] * k[None, :], ell)
    xiell = integrate.simps(integrand, np.log(k))

    return xiell
