# BAO Fitter

This is yet another implementation of a BAO fitter based on Beutler et al. (2017): https://doi.org/10.1093/mnras/stw2373.

We provide a straightforward way to load data, set the model and run the fits. This is not an attempt to replace any of the existing codes, but rather an option for those who require a quick set-up. Please refer to the example notebook to get started.

The fitter works in configuration and Fourier space, but it does not currenlty support a window function in Fourier space (**coming soon**).

It currently uses Zeus (https://github.com/minaskar/zeus) as the default sampler, but other samplers should be relatively simple to add.

#### Requirements:
- numpy
- scipy

Optional:
- zeus-mcmc
- mpi4py
- getdist

#### Acknowledgements:
Thanks to Mariana Vargas Magaña and Sebastien Fromenteau.

The object-oriented approach has been inspired by:
- BARRY: https://github.com/Samreay/Barry/
- Cosmoprimo: https://github.com/cosmodesi/cosmoprimo 
