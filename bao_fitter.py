import numpy as np
import os
import scipy.optimize as op
import zeus
from zeus import ChainManager
from multipoles import *
from power_spectrum import *
from chi_squared import *
from utils import *

class Data:
    def __init__(self,
                 space,
                 label = None,
                 data = None,
                 cov = None,
                 data_file = None,
                 cov_file = None,
                 data_file_type = 'ascii',
                 data_file_cols = (0, 1, 2),
                 cov_file_type = 'ascii',
                 cov_format = '3xN',
                 cov_npoles = 3,
                 ell = (0, 2),
                 recon = None,
                 Sigma_smooth = None,
                 s_min = None,
                 s_max = None,
                 k_min = None,
                 k_max = None):
        
        if space.lower() not in ['configuration', 'fourier']:
            raise Exception("Space must be 'configuration' or 'fourier'")
        
        if space.lower() == 'configuration':
            q = 's'
            pole_labels = [f'xi{l}' for l in ell]
            q_min = s_min
            q_max = s_max
        elif space.lower() == 'fourier':
            q = 'k'
            pole_labels = [f'pk{l}' for l in ell]
            q_min = k_min
            q_max = k_max
        
        self.label = label
        self.space = space
        self.ell = ell
        self.recon = recon
        self.Sigma_smooth = Sigma_smooth
        
        if data:
            array = data
        
        if data_file:
            self.data_file = os.path.abspath(data_file)
            if data_file_type == 'ascii':
                array = np.loadtxt(data_file, usecols=data_file_cols, unpack=True)
            
            elif data_file_type == 'npy':
                array = np.load(data_file).T
                   
        if cov_file:
            self.cov_file = os.path.abspath(cov_file)
            if cov_file_type == 'ascii':
                if cov_format == '3xN':
                    covv = np.loadtxt(cov_file, unpack=True)
                    n_s = int(np.sqrt(len(covv[2]))//cov_npoles)
                    cov = np.reshape(covv[2], (3*n_s, 3*n_s))

                elif cov_format == 'NxN':
                    cov = np.loadtxt(cov_file)
                    n_s = len(cov)//cov_npoles
            elif cov_file_type == 'npy':
                cov = np.load(cov_file)
                n_s = cov.shape[0]//cov_npoles
        
        d = {}
        mask = (array[0]>=q_min) & (array[0]<=q_max)
        d[q] = array[0][mask]
        n_q = len(d[q])
        i_min = np.where(mask)[0][0]
        i_max = np.where(mask)[0][-1]
         
        self.poles = []
        for n in range(len(ell)):
            d[pole_labels[n]] = array[n+1][mask]
            self.poles.append(array[n+1][mask])
            
        self.data = d
        if space.lower() == 'configuration':
            self.s = self.data['s']
        elif space.lower() == 'fourier':
            self.k = self.data['k']

        idx = np.array(sorted(ell))//2
        n_q = len(d[q])
        n_p = len(idx)
        covariance = np.zeros((n_p * n_q, n_p * n_q))
        for jj, n in enumerate(idx):
            for ii, m in enumerate(idx):
                s11 = slice(jj * n_q, (jj + 1) * n_q)
                s12 = slice(ii * n_q, (ii + 1) * n_q)
                s21 = slice(n * n_s + i_min, n * n_s + i_max + 1)
                s22 = slice(m * n_s + i_min, m * n_s + i_max + 1)
                covariance[s11, s12] = cov[s21, s22]
        self.cov = covariance
        self.cov_inv = np.linalg.inv(covariance)
        
    def __call__(self):
            return self.data
        

class Model:
    
    params = {'alpha_par': 1.,
              'alpha_perp': 1.,
              'bias': 1.,
              'beta': 0.,
              'Sigma_par': 0.,
              'Sigma_perp': 0.,
              'Sigma_s': 0.}

    def __init__(self,
                 pk_linear = None,
                 cosmology = None,
                 params = None,
                 recon = None,
                 Sigma_smooth = None):

        self.pk_linear = pk_linear
        self.cosmology = cosmology

        self.params = params if params else self.params
        self.recon = recon
        self.Sigma_smooth = Sigma_smooth

    def power_2D(self, k, mu):
        p2d = power_2D(k, mu, self.pk_linear, recon=self.recon,
                       Sigma_smooth=self.Sigma_smooth, **self.params)
        return p2d
    
    mu = np.linspace(0.0001, 1, 120)
    def pk_poles(self, k, ell=(0,), nmu=120):
        mu = np.linspace(0.0001, 1, nmu) if nmu!= 120 else self.mu
        p2d = self.power_2D(k, mu)
        
        if len(ell) == 1:
            pkell = pk_ell(p2d, mu, ell[0])
        else:
            pkell = []
            for l in ell:
                pkell.append(pk_ell(p2d, mu, l))
        return pkell
    
    def xi_poles(self, s, ell=(0,)):
        k = self.pk_linear[0]
        if len(ell) == 1:
            pkell = self.pk_poles(k, ell=ell)
            xiell = xi_ell(s, ell[0], pkell, k)
        else:
            xiell = []
            pkell = self.pk_poles(k, ell=ell)
            for i, l in enumerate(ell):
                xiell.append(xi_ell(s, l, pkell[i], k))
        return xiell
    

class Fitter:
    
    def __init__(self,
                 data,
                 model,
                 bb_exp,
                 sampler = 'zeus',
                 optimiser = 'L-BFGS-B',
                 fixed_params = None):
        
        self.free_params = ['alpha_par', 'alpha_perp', 'bias', 'beta', 'Sigma_par', 'Sigma_perp', 'Sigma_s']
        self.prior_bounds = {'alpha_par': (0.85, 1.15),
                    'alpha_perp': (0.85, 1.15),
                    'bias': (0.5, 3.),
                    'beta': (0., 1.),
                    'Sigma_par': (0., 12.),
                    'Sigma_perp': (0., 12.),
                    'Sigma_s': (0., 8.)}
        
        self.initial_positions = {}
        for param, value in self.prior_bounds.items():
            self.initial_positions[param] = np.random.uniform(low=value[0], high=value[1])
        
        self.sampler = sampler
        self.bb_exp = bb_exp
        self.optimiser = optimiser
        self.fixed_params = fixed_params
        self.model = model
        self.data = data
        self.space = data.space
        self.model.recon = data.recon
        self.model.Sigma_smooth = data.Sigma_smooth
        
        if fixed_params:
            for param, value in fixed_params.items():
                self.free_params.remove(param)
                self.model.params[param] = value
                del self.prior_bounds[param]
                del self.initial_positions[param]
        
        if self.data.space == 'configuration':
            self.model.poles = self.model.xi_poles
            self.q = self.data.s

        elif self.data.space == 'fourier':
            self.model.poles = self.model.pk_poles
            self.q = self.data.k
    
    def chi2(self, theta):
        for i, param in enumerate(self.free_params):
            self.model.params[param] = theta[i]
        
        c = chi2(self.q, self.data.poles,
                 self.model.poles(self.q, ell=self.data.ell),
                 self.data.cov_inv, self.bb_exp)
        return c
            
    
    def broad_band(self):
        ell = self.data.ell
        if self.space == 'configuration':
            q = self.data.s
            m_poles = self.model.xi_poles(q, ell=ell)
        elif self.space == 'fourier':
            q = self.data.k
            m_poles = self.model.pk_poles(q, ell=ell)
        bb, coeffs = broadband(q, self.data.poles, m_poles, 
                               self.data.cov_inv, self.bb_exp)
        return bb, coeffs
            
    def log_like(self, theta):            
        return -0.5 * self.chi2(theta)
 
    def log_prior(self, theta):
        lp = 0.
        for i, param in enumerate(self.free_params):
            lower_bound = self.prior_bounds[param][0]
            upper_bound = self.prior_bounds[param][1]
            lp += 0. if lower_bound < theta[i] < upper_bound else -np.inf
        return lp
    
    def log_post(self, theta):
        return self.log_like(theta) + self.log_prior(theta)
    
    
    def minimise_chi2(self, x0=None, bounds=None, method=None):
        if not x0:
            x0 = list(self.initial_positions.values())
        if not bounds:
            bounds = list(self.prior_bounds.values())
        if not method:
            method = self.optimiser
        best_fit = op.minimize(self.chi2, x0,
                               method=method, bounds=bounds)
        return best_fit
    
    
    def set_sampler_settings(self, nwalkers=None, nchains=8, epsilon=0.001, nmin=500, nmax=2000, burn_in=0.3):
        if not nwalkers:
            nwalkers = 2 * len(self.free_params)
        self.sampler_settings = {}
        self.sampler_settings['nwalkers'] = nwalkers
        self.sampler_settings['nchains'] = nchains
        self.sampler_settings['R-1'] = epsilon
        self.sampler_settings['nmin'] = nmin
        self.sampler_settings['nmax'] = nmax
        self.sampler_settings['burn-in'] = burn_in
        
    def run_sampler(self, out_path=None):
        if not out_path:
            if self.data.label:
                out_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.data.label.lower() + '/'
            else:
                out_path = os.path.dirname(os.path.realpath(__file__)) + '/' + os.path.basename(__file__) + '/'
            
        bounds = list(self.prior_bounds.values())
        lows = [i[0] for i in bounds]
        highs = [i[1] for i in bounds]
        nwalkers = self.sampler_settings['nwalkers']
        nchains = self.sampler_settings['nchains']
        epsilon = self.sampler_settings['R-1']
        nmin = self.sampler_settings['nmin']
        nmax = self.sampler_settings['nmax']
        burn_in = self.sampler_settings['burn-in']
        log_post = self.log_post
        
        ndim = len(self.free_params)
        start = np.random.uniform(low=lows, high=highs, size=(nwalkers, ndim))
        print('Initial positions: \n', start)
        
        with ChainManager(nchains) as cm:
            rank = cm.get_rank
            if rank == 0:
                if not os.path.isdir(out_path):
                    os.makedirs(out_path)
                    
                    
            cb1 = zeus.callbacks.ParallelSplitRCallback(ncheck=100, nsplits=1, epsilon=epsilon,
                                                       discard=burn_in, chainmanager=cm)
            cb2 = zeus.callbacks.MinIterCallback(nmin=nmin)
            sampler = zeus.EnsembleSampler(nwalkers, ndim, log_post, pool=cm.get_pool)
            sampler.run_mcmc(start, nmax, callbacks=[cb1, cb2]) 
            chain = sampler.get_chain(flat=False, thin=1)
            
            if rank == 0:
                print('R = ', cb1.estimates, flush=True)
            
            chain_path = out_path + 'chain_' + str(rank) + '.npy'
            np.save(chain_path, chain)
            print('Saved file: ' + chain_path)