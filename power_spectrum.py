import numpy as np
import scipy.interpolate as itp
import scipy.integrate as integrate

# Spline
def spline(x, y, x_new):
    s = itp.InterpolatedUnivariateSpline(x, y)
    return s(x_new)

#### FOLLOWING THE CONVENTION FROM BEUTLER ET. AL (2017)

def smooth_hinton2017(ks, pk, degree=13, sigma=1, weight=0.5, **kwargs): # Taken from BARRY
    """ Smooth power spectrum based on Hinton 2017 polynomial method """
    # logging.debug("Smoothing spectrum using Hinton 2017 method")
    log_ks = np.log(ks)
    log_pk = np.log(pk)
    index = np.argmax(pk)
    maxk2 = log_ks[index]
    gauss = np.exp(-0.5 * np.power(((log_ks - maxk2) / sigma), 2))
    w = np.ones(pk.size) - weight * gauss
    z = np.polyfit(log_ks, log_pk, degree, w=w)
    p = np.poly1d(z)
    polyval = p(log_ks)
    pk_smoothed = np.exp(polyval)
    return pk_smoothed

def gaussd(k, mu, S_par, S_perp):
    """ Gaussian damping of the BAO feature """
    gd = np.exp(-(0.5) * ((k ** 2) * (mu ** 2) * (S_par ** 2) + (k ** 2) * (1 - mu ** 2) * (S_perp ** 2)))
    return gd

def D(k, mu, S_s):
    """ Finger of God prefactor """
    f = 1/((1 + 0.5 * (k * mu * S_s) ** 2) **2)
    return f

def kaiser(mu, k, beta, recon=None, Sigma_smooth=None):
    """ Kaiser boost. Iso=True includes the modification implemented in Seo (2016) """
    if recon=='iso':
        ka = (1 + beta * (1 - np.exp(-0.5 * k ** 2 * Sigma_smooth **2)) * (mu ** 2)) ** 2
    else:
        ka = (1 + beta * (mu ** 2)) ** 2
    return ka

# Alcock-Paczynski, change of coordinates and multipoles
# Power spectrum in 2D
def power_2D(k_fid, mu_fid, k_template, pk_lin, alpha_par, alpha_perp, bias, beta, Sigma_par, Sigma_perp, Sigma_s, recon=None, Sigma_smooth=None):
    F = alpha_par / alpha_perp

    # define real coordinates
    k_p = (k_fid[:, None] / alpha_perp) * np.sqrt( 1 + mu_fid[None, :] ** 2 * (1 / (F ** 2) - 1) )
    mu_p = (mu_fid / F) / np.sqrt( 1 + mu_fid ** 2 * (1 / (F ** 2) - 1) )

    # no-wiggles power spectrum
    pnw = smooth_hinton2017(k_template, pk_lin)
    pk_int = spline(np.log(k_template), pk_lin, np.log(k_p))
    pnw_int = spline(np.log(k_template), pnw, np.log(k_p))

    # construct power spectrum in 2D
    pdw = np.multiply((pk_int - pnw_int), gaussd(k_p, mu_p, Sigma_par, Sigma_perp)) + pnw_int
    p2d = pdw * np.multiply( D(k_p, mu_p, Sigma_s), kaiser(mu_p, k_p, beta, Sigma_smooth, recon) )
    p2d *= bias ** 2
    p2d *= 1 / (alpha_perp ** 2 * alpha_par)

    return p2d