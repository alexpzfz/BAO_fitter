import numpy as np
import scipy.interpolate as itp
import scipy.optimize as op
import scipy.integrate as integrate
import sys
import os
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
best_fit = op.minimize(chi2, [1, 0.35, 1, 1, 4, 2.5, 3],
                       args=(data, cov_inv, [k, ps_lin], mu, sigma_r, iso, bb_exp, space),
                       bounds=[(1, 10), (0, 5), (0.8, 1.2), (0.8, 1.2), (0, 12), (0, 12), (0, 8)])

# get broad band coefficients
b, beta, alpha_par, alpha_per, S_par, S_per, S_s = best_fit.x
if space=='config':
    pt = power_spectrum_template(k, mu, ps_lin, S_par, S_per, S_s, sigma_r, b, beta, iso)
    model = two_point_cf_template(data[0], k, mu, pt, alpha_par, alpha_per)

elif space=='fourier':
    model = ps_multipoles_template(data[0], k, mu, ps_lin, S_par, S_per, S_s,
                                   sigma_r, b, beta, iso, alpha_par, alpha_per)

coeffs = broadband(data, model, cov_inv, bb_exp)[2]

# save as a dictionary
data_dict = {}
data_dict['label'] = label
data_dict['multipoles'] = data
data_dict['covariance'] = cov_matrix
data_dict['Sigma_r'] = sigma_r
data_dict['iso'] = iso
data_dict['best_fit_out'] = best_fit
data_dict['best_fit_values'] = {'b':b, 'beta':beta, 'alpha_par':alpha_par, 
                                'alpha_per':alpha_per, 'Sigma_par':S_par, 
                                'Sigma_per':S_per, 'Sigma_fog':S_s}
data_dict['broad_band_exp'] = bb_exp
data_dict['broad_band_coeff'] = np.split(coeffs, 2)
with open(out_path + label.lower() + '.dict', 'wb') as file:
    pickle.dump(data_dict, file)

# save loglikelihood for cronus
fitter_path = os.path.dirname(os.path.realpath(__file__))
loglike_path = label.lower() + '_loglikelihood.py'
with open(loglike_path, "w") as file:
    file.write(f"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
fitter_path = '{fitter_path}'
sys.path.append(fitter_path)
import numpy as np
from chi_squared import *
from truncate_covariance import *

# multipoles
mul_path = '{mul_path}'
q_min = {q_min}
q_max = {q_max}
data = np.loadtxt(mul_path, usecols=(0,1,2), unpack=True)
q_mask = (q_min<=data[0]) & (data[0]<=q_max)
data = data[:, q_mask]

# covariance matrix
cov_path = '{cov_path}'
cov_matrix = truncate(cov_path, q_min, q_max)
cov_inv = np.linalg.inv(cov_matrix)

# linear template
ps_lin_path = '{ps_lin_path}'
linear_template = np.loadtxt(ps_lin_path, unpack=True)

# mu
n_mu = {n_mu}
mu = np.linspace(0.001, 1., n_mu)

# rest of the parameters
S_r = {sigma_r}
iso = {iso}
bb_exp = {bb_exp}
space = '{space}'

# likelihood function
def log_like(theta):
    l = loglike(theta, data, cov_inv, linear_template,
                mu, S_r, iso, bb_exp, space)
    return l
""")

# save .yaml for chronus
yaml_path = label.lower() + '.yaml'
with open(yaml_path, 'w') as file:
    file.write(f"""
Likelihood:
  path: {loglike_path}
  function: log_like

Parameters:
  b:
    prior:
      type: uniform
      min: 0.5
      max: 3
  beta:
    prior:
      type: uniform
      min: 0
      max: 2
  alpha_par:
    prior:
      type: uniform
      min: 0.9
      max: 1.1
  alpha_per:
    prior:
      type: uniform
      min: 0.9
      max: 1.1
  Sigma_par:
    fixed: {S_par}
  Sigma_per:
    fixed: {S_per}
  S_fog:
    fixed: {S_s}

Sampler:
  name: zeus
  ndim: 7
  nwalkers: 14
  nchains: 2
  initial: ellipse

Diagnostics:
  Gelman-Rubin:
    use: True
    epsilon: 0.05
  Autocorrelation:
    use: True
    nact: 20
    dact: 0.03

Output:
  directory: {out_path}
""")
