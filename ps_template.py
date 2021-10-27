import numpy as np
import scipy.interpolate as itp
import scipy.integrate as integrate


#### FOLLOWING THE CONVENTION FROM BEUTLER ET. AL (2017)

def smooth_hinton2017(ks, pk, degree=13, sigma=1, weight=0.5, **kwargs):
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

def gaussd(k, mu, S_per, S_par):
    """ Gaussian damping of the BAO feature """
    gd = np.exp(-(0.5)*((k**2)*(mu**2)*(S_par**2) + (k**2)*(1-mu**2)*(S_per**2)))
    return gd

def F(k, mu, S_s):
    """ Finger of God prefactor """
    f = 1/((1 + 0.5*(k*mu*S_s)**2)**2) 
    return f

def kaiser(mu, k, beta, S_r, iso=True):
    """ Kaiser boost. Iso=True includes the modification implemented in Seo (2016) """
    if iso:
        ka = (1 + beta*(1 - np.exp(-0.5 * k**2 * S_r**2))*(mu**2))**2
    else:
        ka = (1 + beta*(mu**2))**2
    return ka

def power_spectrum_template(k, mu, pk_lin, S_per, S_par, S_s, S_r, b, beta, iso=True):
    """ Power spectrum in (k, mu) space """
    pnw = smooth_hinton2017(k, pk_lin)
    pdw = np.multiply((pk_lin - pnw)[:,None], gaussd(k[:,None], mu[None,:], S_per, S_par)) + pnw[:,None]
    pt = pdw * np.multiply(F(k[:,None], mu[None,:], S_s), kaiser(mu[None,:], k[:, None], beta, S_r, iso))
    pt *= b**2
    return pt
