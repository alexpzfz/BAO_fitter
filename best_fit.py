import numpy as np
import scipy.interpolate as itp
import scipy.optimize as op
import scipy.integrate as integrate
import sys
import pickle
from ps_template import *
from multipoles import *
from chi_squared import *
from truncate_covariance import *

# read parameters from file
with open(sys.argv[1], 'r') as file:
    lines = file.readlines()
label = lines[0].split()[1]
mul_path = lines[1].split()[1]
cov_path = lines[2].split()[1]
out_path = lines[3].split()[2]
ps_lin_path = lines[4].split()[3]
sigma_r = float(lines[5].split()[1])
iso = bool(int(lines[6].split()[2]))
space = lines[7].split()[1]
q_min = float(lines[8].split()[1]) # q means either s or k
q_max = float(lines[9].split()[1])
n_mu = int(lines[10].split()[1])
bb_exp = list(map(int, lines[11].split(sep=': ')[1].split(sep=', ')))

# Linear template and mu
k, ps_lin = np.loadtxt(ps_lin_path, unpack=True)
mu = np.linspace(0.001, 1., n_mu)

# multipoles
#data = np.loadtxt(mul_path, usecols=(0,3,4), unpack=True)
data = np.loadtxt(mul_path, usecols=(0,1,2), unpack=True)
q_mask = (q_min<=data[0]) & (data[0]<=q_max)
data = data[:, q_mask]

# covariance matrix
cov_matrix = truncate(cov_path, q_min, q_max)
cov_inv = np.linalg.inv(cov_matrix)

# minimise
best_fit = op.minimize(chi2, [1, 0.35, 1, 1, 4, 2.5, 3], args=(data, cov_inv, [k, ps_lin], mu, sigma_r, iso, bb_exp, None, space), bounds=[(1, 10), (0, 5), (0.8, 1.2), (0.8, 1.2), (0, 12), (0, 12), (0, 8)])

# get broad band coefficients
b, beta, alpha_par, alpha_per, S_par, S_per, S_s = best_fit.x
if space=='config':
    pt = power_spectrum_template(k, mu, ps_lin, S_par, S_per, S_s, sigma_r, b, beta, iso)
    model = two_point_cf_template(data[0], k, mu, pt, alpha_par, alpha_per)

elif space=='fourier':
    model = ps_multipoles_template(data[0], k, mu, ps_lin, S_par, S_per, S_s, sigma_r, b, beta, iso, alpha_par, alpha_per)

coeffs = broadband(data, model, cov_inv, bb_exp)[2]

# save as a dictionary
data_dict = {}
data_dict['label'] = label
data_dict['multipoles'] = data
data_dict['covariance'] = cov_matrix
data_dict['Sigma_r'] = sigma_r
data_dict['iso'] = iso
data_dict['best_fit_out'] = best_fit
data_dict['best_fit_values'] = {'b':b, 'beta':beta, 'alpha_par':alpha_par, 'alpha_per':alpha_per, 'Sigma_par':S_par, 'Sigma_per':S_per, 'Sigma_fog':S_s}
data_dict['broad_band_exp'] = bb_exp
data_dict['broad_band_coeff'] = np.split(coeffs, 2)
with open(out_path + label.lower() + '.dict', 'wb') as file:
    pickle.dump(data_dict, file)
