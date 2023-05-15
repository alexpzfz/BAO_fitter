import numpy as np
import os, sys
sys.path.append('../')
from bao_fitter import Data, Model, Fitter
import matplotlib.pyplot as plt

d = Data(space='fourier', # can be fourier or configuration
         data_file='pk_multipoles.txt', 
         data_file_type='ascii', # a text file in this case, can use npy
         data_file_cols=(0,1,2),
         cov_file='covariance.npy', 
         cov_file_type='npy', # npy but can be ascii
         cov_npoles=2, # number of multipoles in the covariance file
         ell=(0,2), k_min=0.02, k_max=0.2, recon='recsym')

pk_lin = np.loadtxt('linearPk.txt', unpack=True)
m = Model(pk_linear=pk_lin)

f = Fitter(data=d,
           model=m,
           bb_exp=(-2, -1, 0, 1, 2), # exponents of the broad-band terms to use
           fixed_params={'Sigma_par':4., 'Sigma_perp':6., 'Sigma_s':0., 'beta':0.3}, # you can specify if you want some of the parameters to be fixed
           )

f.set_sampler_settings(nchains=2,
                       epsilon=0.001, # R-1 for Gelman-Rubin
                       nmin=1000,
                       nmax=2000,
                      )

f.run_sampler(out_path='./')
