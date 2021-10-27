import numpy as np
import scipy.interpolate as itp
import scipy.integrate as integrate

# Legendre polyonomials
def lp2(x):
    l_2 = 0.5*(3*x**2 - 1)
    return l_2

def lp4(x):
    l_4 = 0.125*(35*x**4 - 30*x**2 + 3)
    return l_4

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

def ps_multipoles_template(k_data, k_template, mu_fid, pk_lin, S_per, S_par, S_s, S_r, b, beta, iso, alpha_par, alpha_per):
    
    # prime coordinates
    k_p = k_template
    mu_p = mu_fid/(alpha_par * g_inv(mu_fid, alpha_par, alpha_per)) 
    # mu_fid must be between 0 and 1
    
    # evaluate template
    ps_template = power_spectrum_template(k_p, mu_p, pk_lin, S_per, S_par, S_s, S_r, b, beta, iso)
    
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
