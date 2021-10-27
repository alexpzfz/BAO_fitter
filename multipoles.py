import numpy as np
import scipy.interpolate as itp
import scipy.integrate as integrate
from ps_template.py import *

# Legendre polyonomials
def lp2(x):
    l_2 = 0.5*(3*x**2 - 1)
    return l_2

def lp4(x):
    l_4 = 0.125*(35*x**4 - 30*x**2 + 3)
    return l_4

# Spherical Bessel functions
def j0(x):
    j_0 = np.sin(x)/x
    return j_0

def j2(x):
    j_2 = (3/(x**2) - 1)*(np.sin(x)/x) - 3*np.cos(x)/(x**2)
    return j_2

def j4(x):
    j_4 = (1/(x**5))*(5*x*(2*x**2 - 21)*np.cos(x) + (x**4 - 45*x**2 + 105)*np.sin(x))
    return j_4

# Spline
def spline(x, y, x_new):
    s = itp.InterpolatedUnivariateSpline(x, y)
    return s(x_new)

#### FOURIER SPACE MULTIPOLES ####

# Alcok-Paczynski, change of coordinates and multipoles

# k_fid = k_p * g(mu_p)
# mu_fid = mu_p * alpha_par/g(mu_p)
# g(mu_p)= alpha_per * (1 + mu_p**2 * (F**2 -1))**(1/2), F = alpha_par/alpha_per

# g(mu_p=0) =  alpha_per
# g(mu_p=1,-1) = alpha_par

def g(mup, alpha_par, alpha_per):
    F = alpha_par/alpha_per
    return alpha_per * (1 + mup**2 * (F**2 -1))**(0.5)

def g_inv(mu, alpha_par, alpha_per):
    F = alpha_par/alpha_per
    return (1/alpha_per) * (1 + mu**2 * (1/F**2 -1))**(0.5)

def ps_multipoles_template(k_data, k_template, mu_fid, pk_lin, S_par, S_per, S_s, S_r, b, beta, iso, alpha_par, alpha_per):
    
    # prime coordinates
    k_p = k_template
    mu_p = mu_fid/(alpha_par * g_inv(mu_fid, alpha_par, alpha_per)) 
    # mu_fid must be between 0 and 1
    
    # evaluate template
    ps_template = power_spectrum_template(k_p, mu_p, pk_lin, S_par, S_per, S_s, S_r, b, beta, iso)
    
    # delimiting k_fiducial
    g_arr = g(mu_p, alpha_par, alpha_per)
    g_min = np.min([alpha_par, alpha_per])
    g_max = np.max([alpha_par, alpha_per])
    k_p_min = k_p[0]
    k_p_max = k_p[-1]
    k_fid = np.geomspace(g_max*k_p_min, g_min*k_p_max, len(k_p))
    
    # Power spectrum evaluated in fiducial coordinates
    ps_fid = np.zeros((len(k_fid), len(mu_fid)))
    for i in range(len(mu_p)):
        g_loc = g_arr[i]
        k_fid_loc = k_p * g_loc
        ps_loc = ps_template[:, i]
        ps_fid_log = spline(k_fid_loc, np.log(ps_loc), k_fid)
        ps_fid[:,i] = np.exp(ps_fid_log)
    
    
    p0_complete = 1/(alpha_per**2 * alpha_par) * integrate.simps(ps_fid, mu_fid , axis=1)
    p2_complete = 5/(alpha_per**2 * alpha_par) * integrate.simps(ps_fid * lp2(mu_fid), mu_fid, axis=1)
    
    kp0 = spline(k_fid, k_fid*p0_complete, k_data)
    kp2 = spline(k_fid, k_fid*p2_complete, k_data)

    p0 = kp0/k_data
    p2 = kp2/k_data
    
    return p0, p2


#### CONFIGURATION SPACE MULTIPOLES ####

def two_point_cf_template(r, k, mu, pt, alpha_par, alpha_per):
    # monopole
    p0t = (0.5) * np.trapz(pt, mu,1)
    f0 = (0.5/((np.pi)**2))*(k**3)*p0t * np.exp(-k**2)
    xi0t = np.trapz(f0*j0(r[:,None] * k[None,:]), np.log(k), 1)

    # quadrupole
    p2t = (5*0.5) * np.trapz(pt * lp2(mu), mu, 1)
    f2 = -(0.5/((np.pi)**2))*(k**3)*p2t * np.exp(-k**2)
    xi2t = np.trapz(f2*j2(r[:,None] * k[None,:]), np.log(k), 1)

    # hexadecapole
    p4t = (9*0.5) * np.trapz(pt * lp4(mu), mu, 1)
    f4 = (0.5/((np.pi)**2))*(k**3)*p4t * np.exp(-k**2)
    xi4t = np.trapz(f4*j4(r[:,None] * k[None,:]), np.log(k), 1)
    
    # Alcock Paczynski
    alpha = (alpha_par)**(1/3) * (alpha_per)**(2/3)
    epsilon = (alpha_par/alpha_per)**(1/3) - 1
    mu_f2 = mu**2
    r_fid2 = r**2
    mu_obs = np.sqrt(1/(1 + (1/mu_f2 - 1)/((epsilon + 1)**6)))
    mu_obs2 = 1/(1 + (1/mu_f2 - 1)/((epsilon + 1)**6))
    r_obs = r[:, None] * alpha * np.sqrt((epsilon + 1)**4 * mu_f2[None,:] + (1 - mu_f2[None,:])/((1 + epsilon)**2))
    r_obs2 = r_obs**2
    l2_obs = lp2(mu_obs)
    l4_obs = lp4(mu_obs)
    xi0_obs = spline(r, r_fid2 * xi0t, r_obs)/r_obs2
    xi2_obs = spline(r, r_fid2 * xi2t, r_obs)/r_obs2 * l2_obs[None,:]
    xi4_obs = spline(r, r_fid2 * xi4t, r_obs)/r_obs2 * l4_obs[None,:]
    xi_obs = xi0_obs + xi2_obs + xi4_obs
    xi0 = 0.5 * np.trapz(xi_obs, mu, 1)
    xi2 = 5 * 0.5 * np.trapz(xi_obs * lp2(mu)[None,:], mu, 1)
    # xi4 = 9.* 0.5 * np.trapz(xi_obs * lp4(mu)[None,:], mu, 1)

    return xi0, xi2
